"""
Stage 2 — 하드웨어 프로빙 (HW Probe).

psutil, py-cpuinfo, pynvml을 사용해 시스템 사양을 수집하고
HWCaps 스냅샷을 반환한다.

macOS(Apple Silicon), Linux(NVIDIA/AMD), Windows(NVIDIA) 환경
모두 지원하며, 특정 라이브러리가 없으면 안전하게 폴백한다.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import psutil

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import CPUInfo, GPUInfo, HWCaps

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def _probe_cpu() -> CPUInfo:
    """CPU 정보 수집 (py-cpuinfo 우선, 폴백 시 psutil)."""
    try:
        import cpuinfo  # py-cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        return CPUInfo(
            brand=info.get("brand_raw", platform.processor()),
            cores_physical=psutil.cpu_count(logical=False) or 1,
            cores_logical=psutil.cpu_count(logical=True) or 1,
            has_avx2="avx2" in flags,
            has_avx512=any(f.startswith("avx512") for f in flags),
        )
    except ImportError:
        # py-cpuinfo 미설치 시 psutil로 폴백
        return CPUInfo(
            brand=platform.processor(),
            cores_physical=psutil.cpu_count(logical=False) or 1,
            cores_logical=psutil.cpu_count(logical=True) or 1,
        )


def _probe_nvidia_via_smi() -> list[GPUInfo]:
    """`nvidia-smi --query-gpu` fallback — pynvml NVML init 실패 시 사용.

    aarch64 NVIDIA Open Kernel Module 등 일부 환경에서 pynvml은 'Not Supported'.
    nvidia-smi는 CLI라 NVML 라이브러리 ABI에 덜 민감.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return []

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception as e:
        log.debug("nvidia-smi 호출 실패", extra={"extra": {"err": str(e)}})
        return []

    def _safe_float(s: str) -> float:
        """nvidia-smi는 unified memory / 일부 SKU에서 'Not Supported' / '[N/A]' 반환."""
        s = s.strip()
        if not s or s.startswith("[") or s.lower().startswith("not "):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
        except (ValueError, IndexError):
            continue
        vram_mib = _safe_float(parts[2]) if len(parts) > 2 else 0.0
        temp = _safe_float(parts[3]) if len(parts) > 3 else 0.0
        util = _safe_float(parts[4]) if len(parts) > 4 else 0.0
        gpus.append(GPUInfo(
            index=idx, name=name, vendor="nvidia",
            vram_gb=round(vram_mib / 1024, 2),  # 0.0이면 unified memory (RAM 공유)
            compute_capability="",  # nvidia-smi에 표준 컬럼 없음
            utilization_pct=util,
            temperature_c=temp,
        ))
    return gpus


def _probe_nvidia_gpus() -> list[GPUInfo]:
    """NVIDIA GPU 정보 수집 — pynvml 우선, 실패 시 nvidia-smi fallback."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GPUInfo] = []

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            # pynvml 버전에 따라 bytes or str 반환
            if isinstance(name, bytes):
                name = name.decode()

            # CUDA compute capability
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cc = f"{major}.{minor}"
            except Exception:
                cc = ""

            # 온도
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                ))
            except Exception:
                temp = 0.0

            # 사용률
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_pct = float(util.gpu)
            except Exception:
                util_pct = 0.0

            gpus.append(GPUInfo(
                index=i,
                name=name,
                vendor="nvidia",
                vram_gb=round(mem.total / (1024 ** 3), 2),
                compute_capability=cc,
                utilization_pct=util_pct,
                temperature_c=temp,
            ))

        pynvml.nvmlShutdown()
        if gpus:
            return gpus
        # NVML init 성공했지만 0개 — fallback 시도
        log.debug("NVML 0개 GPU 보고 — nvidia-smi fallback 시도")
        return _probe_nvidia_via_smi()

    except ImportError:
        log.debug("pynvml 미설치 — nvidia-smi fallback")
        return _probe_nvidia_via_smi()
    except Exception as e:
        # NVML init 실패 (예: aarch64 Open Kernel Module) — nvidia-smi로 fallback
        log.info(
            "pynvml 사용 불가 — nvidia-smi fallback",
            extra={"extra": {"err": str(e)}},
        )
        return _probe_nvidia_via_smi()


def _probe_apple_gpu(unified_mem_gb: float) -> list[GPUInfo]:
    """Apple Silicon GPU 정보 수집 (system_profiler 사용)."""
    if sys.platform != "darwin":
        return []

    # system_profiler로 GPU 이름 추출
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        import json
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [{}])
        gpu_name = displays[0].get("sppci_model", "Apple GPU") if displays else "Apple GPU"
    except Exception:
        gpu_name = "Apple GPU"

    return [GPUInfo(
        index=0,
        name=gpu_name,
        vendor="apple",
        # Apple Silicon은 Unified Memory — CPU와 공유
        vram_gb=unified_mem_gb,
        compute_capability="",
    )]


def _probe_amd_gpus() -> list[GPUInfo]:
    """AMD ROCm GPU 정보 수집 (pyrsmi 사용, 선택적)."""
    try:
        from pyrsmi import rocml  # type: ignore[import]

        rocml.smi_initialize()
        count = rocml.smi_get_device_count()
        gpus: list[GPUInfo] = []

        for i in range(count):
            name = rocml.smi_get_device_name(i)
            vram = rocml.smi_get_device_vram_usage(i)  # bytes
            gpus.append(GPUInfo(
                index=i,
                name=name,
                vendor="amd",
                vram_gb=round(vram / (1024 ** 3), 2),
            ))

        rocml.smi_shutdown()
        return gpus
    except ImportError:
        return []
    except Exception as e:
        log.warning("AMD GPU 감지 실패", extra={"extra": {"err": str(e)}})
        return []


def _check_torch_backends() -> tuple[bool, bool]:
    """torch 레벨 CUDA / MPS 가용 여부 확인.

    torch 미설치 시:
      - macOS + Apple Silicon (arm64) → MPS capable로 간주 (학습 패키지 설치 시 자동 활성)
      - 그 외 → False
    """
    import platform
    is_macos_arm = sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        # MPS: is_available() 우선, 실패 시 is_built() fallback (torch wheel이 MPS 포함하지만 device 없음 케이스)
        mps_ok = False
        if sys.platform == "darwin" and hasattr(torch.backends, "mps"):
            try:
                mps_ok = torch.backends.mps.is_available()
            except Exception:
                mps_ok = False
            if not mps_ok and hasattr(torch.backends.mps, "is_built"):
                try:
                    mps_ok = torch.backends.mps.is_built() and is_macos_arm
                except Exception:
                    pass
        return cuda_ok, mps_ok
    except ImportError:
        # torch 미설치 — macOS Apple Silicon이면 MPS capable로 표시 (UI/규칙 매칭용)
        # 학습 시점에 torch 설치 필수. is_macos_arm 자체는 환경 진단 신호.
        return False, is_macos_arm


def _estimate_network_speed() -> float:
    """
    네트워크 속도 추정 (Mbps).

    psutil로 초기·1초 후 바이트 수를 비교. 정확도보다 ETA 추정용.
    실패 시 0.0 반환.
    """
    try:
        import time

        s1 = psutil.net_io_counters()
        time.sleep(0.5)
        s2 = psutil.net_io_counters()
        bytes_per_sec = (s2.bytes_sent + s2.bytes_recv - s1.bytes_sent - s1.bytes_recv) / 0.5
        return round(bytes_per_sec * 8 / 1e6, 2)  # Mbps
    except Exception:
        return 0.0


class HWProber:
    """
    Stage 2: 시스템 하드웨어 사양 수집기.

    책임:
    - OS / CPU / GPU / RAM / Disk / Network 정보 수집
    - HWCaps 스냅샷 반환
    - torch 백엔드 최종 확인
    """

    def probe(self) -> HWCaps:
        """
        전체 하드웨어 프로빙 실행.

        Returns
        -------
        HWCaps
            수집된 하드웨어 능력 스냅샷.

        Raises
        ------
        WFPError(WFP-0201)
            기본 시스템 정보 수집 자체가 실패한 경우.
        """
        log.info("Stage 2: HW 프로빙 시작", extra={"stage": 2})

        try:
            os_name = self._detect_os()
            cpu = _probe_cpu()
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
            disk_free_gb = round(
                psutil.disk_usage("/").free / (1024 ** 3), 2
            )
        except Exception as e:
            raise WFPError(
                code=WFPErrorCode.HW_PROBE_FAILED,
                message="기본 시스템 정보 수집 실패",
                stage=2,
                detail=str(e),
            ) from e

        # Unified Memory (Apple Silicon)
        unified_mem_gb = ram_gb if os_name == "darwin" else 0.0

        # GPU 수집 — 각 벤더별 독립 시도
        gpus: list[GPUInfo] = []

        if os_name == "darwin":
            gpus = _probe_apple_gpu(unified_mem_gb)
        else:
            gpus = _probe_nvidia_gpus() or _probe_amd_gpus()

        # torch 최종 검증
        cuda_available, mps_available = _check_torch_backends()

        # torch가 없어도 감지된 GPU 정보는 유지
        max_vram = max((g.vram_gb for g in gpus), default=0.0)

        # 네트워크 속도 추정
        network_mbps = _estimate_network_speed()

        caps = HWCaps(
            os=os_name,
            cpu=cpu,
            gpus=gpus,
            ram_gb=ram_gb,
            disk_free_gb=disk_free_gb,
            network_mbps=network_mbps,
            max_vram_gb=max_vram,
            unified_memory_gb=unified_mem_gb,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )

        log.info(
            "Stage 2: HW 프로빙 완료",
            extra={
                "stage": 2,
                "extra": {
                    "os": os_name,
                    "ram_gb": ram_gb,
                    "disk_free_gb": disk_free_gb,
                    "gpu_count": len(gpus),
                    "max_vram_gb": max_vram,
                    "cuda": cuda_available,
                    "mps": mps_available,
                },
            },
        )

        return caps

    @staticmethod
    def _detect_os() -> str:
        """플랫폼 → 'linux' | 'windows' | 'darwin' 반환."""
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        if system == "windows":
            return "windows"
        if system == "darwin":
            return "darwin"
        return system
