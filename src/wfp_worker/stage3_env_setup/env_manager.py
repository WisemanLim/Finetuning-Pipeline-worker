"""
Stage 3 — 환경 설정 (Env Setup).

Python 버전, 필수 라이브러리 설치 여부, GPU 드라이버,
OS별 권고 사항을 검증하고 JobSpec을 갱신한다.

실제 설치 자동화는 `uv sync` 명령을 subprocess로 호출.
드라이버 부재 시 경고만 발생 — 강제 중단은 하지 않음(CPU 폴백 가능).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import HWCaps, JobSpec, JobState

log = get_logger(__name__)

# 필수 패키지 (import 이름 → pip/uv 패키지 이름)
_REQUIRED_PACKAGES: dict[str, str] = {
    "psutil": "psutil",
    "fastapi": "fastapi",
    "yaml": "pyyaml",
    "pydantic": "pydantic",
}

# 훈련 시 필요한 패키지 (train 옵션 그룹)
_TRAIN_PACKAGES: dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "accelerate": "accelerate",
    "datasets": "datasets",
}

# Python 최소 버전
_MIN_PYTHON = (3, 12)


@dataclass
class EnvCheckResult:
    """환경 검사 결과 요약."""

    python_ok: bool
    python_version: str
    missing_required: list[str]
    missing_train: list[str]
    cuda_version: str
    driver_warnings: list[str]
    os_warnings: list[str]

    @property
    def is_ready(self) -> bool:
        """필수 패키지가 모두 있으면 준비 완료."""
        return self.python_ok and not self.missing_required


class EnvManager:
    """
    Stage 3: 환경 검증 및 설정.

    책임:
    - Python 버전 검증
    - 필수·선택 라이브러리 존재 확인
    - GPU 드라이버(CUDA/MPS) 상태 확인
    - OS별 권고 메시지 생성
    - JobSpec.env_ready / python_version / cuda_version 갱신
    """

    def setup(self, spec: JobSpec) -> EnvCheckResult:
        """
        환경 검증 실행 및 JobSpec 갱신.

        Parameters
        ----------
        spec:
            Stage 2까지 채워진 JobSpec.
            hw_caps가 있으면 드라이버 검증에 활용.

        Returns
        -------
        EnvCheckResult
            검사 결과 요약. 호출자가 경고를 사용자에게 전달할 수 있음.

        Raises
        ------
        WFPError(WFP-0302)
            필수 패키지 중 하나라도 누락된 경우.
        WFPError(WFP-0301)
            Python 버전이 최소 요구 사항 미달인 경우.
        """
        log.info("Stage 3: 환경 검사 시작", extra={"stage": 3, "job_id": str(spec.job_id)})

        result = EnvCheckResult(
            python_ok=False,
            python_version="",
            missing_required=[],
            missing_train=[],
            cuda_version="",
            driver_warnings=[],
            os_warnings=[],
        )

        # Python 버전 확인
        result.python_version = platform.python_version()
        major, minor = sys.version_info.major, sys.version_info.minor
        result.python_ok = (major, minor) >= _MIN_PYTHON
        if not result.python_ok:
            raise WFPError(
                code=WFPErrorCode.PYTHON_VERSION_MISMATCH,
                message=f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ 필요. 현재: {result.python_version}",
                stage=3,
            )

        # 필수 패키지 확인
        result.missing_required = self._check_imports(_REQUIRED_PACKAGES)
        if result.missing_required:
            raise WFPError(
                code=WFPErrorCode.MISSING_DEPENDENCY,
                message=f"필수 패키지 미설치: {result.missing_required}",
                stage=3,
                detail="uv sync 또는 pip install로 설치하세요.",
            )

        # 학습 패키지 확인 (경고만)
        result.missing_train = self._check_imports(_TRAIN_PACKAGES)
        if result.missing_train:
            log.warning(
                "학습 패키지 미설치 — 실제 학습 불가",
                extra={"stage": 3, "extra": {"missing": result.missing_train}},
            )

        # GPU 드라이버 확인
        hw_caps: HWCaps | None = spec.hw_caps
        result.cuda_version = self._detect_cuda_version()
        result.driver_warnings = self._check_driver(hw_caps)
        result.os_warnings = self._check_os_specifics(hw_caps)

        # JobSpec 갱신
        spec.env_ready = result.is_ready
        spec.python_version = result.python_version
        spec.cuda_version = result.cuda_version
        spec.update_stage(stage=3, state=JobState.RUNNING)

        log.info(
            "Stage 3: 환경 검사 완료",
            extra={
                "stage": 3,
                "job_id": str(spec.job_id),
                "extra": {
                    "python": result.python_version,
                    "cuda": result.cuda_version,
                    "env_ready": result.is_ready,
                    "driver_warnings": result.driver_warnings,
                },
            },
        )

        return result

    # ─────────────────────────────────────────
    # 내부 유틸
    # ─────────────────────────────────────────

    @staticmethod
    def _check_imports(packages: dict[str, str]) -> list[str]:
        """import 시도 후 실패한 패키지 이름(pip 이름) 목록 반환."""
        missing = []
        for import_name, pip_name in packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pip_name)
        return missing

    @staticmethod
    def _detect_cuda_version() -> str:
        """nvcc 또는 nvidia-smi로 CUDA 버전 감지. 없으면 빈 문자열."""
        # nvcc 우선
        nvcc = shutil.which("nvcc")
        if nvcc:
            try:
                out = subprocess.check_output(
                    ["nvcc", "--version"], text=True, timeout=5
                )
                for line in out.splitlines():
                    if "release" in line.lower():
                        # "Cuda compilation tools, release 12.1, V12.1.66"
                        parts = line.split("release")
                        if len(parts) > 1:
                            return parts[1].split(",")[0].strip()
            except Exception:
                pass

        # nvidia-smi 폴백
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True, timeout=5,
                )
                return f"driver:{out.strip().splitlines()[0]}"
            except Exception:
                pass

        return ""

    @staticmethod
    def _check_driver(caps: HWCaps | None) -> list[str]:
        """GPU 드라이버 경고 메시지 목록 반환."""
        warnings: list[str] = []
        if caps is None:
            return warnings

        has_nvidia = any(g.vendor == "nvidia" for g in caps.gpus)
        has_apple = any(g.vendor == "apple" for g in caps.gpus)

        if has_nvidia and not caps.cuda_available:
            warnings.append(
                "NVIDIA GPU 감지되었으나 CUDA를 사용할 수 없습니다. "
                "드라이버 또는 torch CUDA 빌드를 확인하세요."
            )

        if has_apple and not caps.mps_available:
            warnings.append(
                "Apple Silicon 감지되었으나 MPS 백엔드 비활성. "
                "torch>=2.0 MPS 빌드가 필요합니다."
            )

        if not caps.gpus:
            warnings.append("GPU가 감지되지 않았습니다. CPU 학습은 매우 느립니다.")

        return warnings

    @staticmethod
    def _check_os_specifics(caps: HWCaps | None) -> list[str]:
        """OS별 권고 사항 메시지 반환."""
        warnings: list[str] = []
        if caps is None:
            return warnings

        if caps.os == "windows":
            # Windows + NVIDIA 조합이 아니면 WSL2 권고
            has_nvidia = any(g.vendor == "nvidia" for g in caps.gpus)
            if not has_nvidia:
                warnings.append(
                    "Windows 환경에서 GPU 학습은 WSL2 + Ubuntu를 권장합니다."
                )

        if caps.os == "darwin":
            # bitsandbytes macOS 미지원 — mlx-lm/unsloth 안내
            warnings.append(
                "macOS: bitsandbytes 미지원. QLoRA는 unsloth(MPS) 또는 mlx-lm을 사용합니다."
            )

        return warnings
