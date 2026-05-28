"""
환경 진단 + 자동 설치 매트릭스.

OS × GPU 조합에 따라 권장 패키지가 다름:
  darwin + apple   → torch MPS, mlx-lm(옵션)
  linux  + nvidia  → torch CUDA + bitsandbytes
  linux  + amd     → torch ROCm
  windows + nvidia → torch CUDA (Windows wheel)
  any + none(CPU)  → torch CPU

두 진입점:
  1. CLI: `python -m wfp_worker.env_check [--install]`
     호스트 dev 환경 (uv venv) 또는 Dockerfile 빌드 시 사용
  2. REST: `/admin/env/status` + `/admin/env/install`
     관리자가 대시보드 진입 시 호출. subprocess로 uv pip install 실행.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import HWCaps

log = get_logger(__name__)

# ─────────────────────────────────────────
# 환경 프로파일 정의
# ─────────────────────────────────────────

# import 이름 → 사용자 표시 이름
_BASE_TRAIN_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "evaluate": "evaluate",
}

# GPU 전용
_NVIDIA_EXTRA = {"bitsandbytes": "bitsandbytes"}

# 프로파일 — 환경별 권장 설치 명령
# (uv 기준; pip 기본 호환)
PROFILES: dict[str, dict] = {
    "nvidia": {
        "label": "NVIDIA CUDA",
        "extras": "train_gpu",
        "torch_index": "https://download.pytorch.org/whl/cu121",
        "packages": {**_BASE_TRAIN_PACKAGES, **_NVIDIA_EXTRA},
    },
    "apple": {
        "label": "Apple Silicon (MPS)",
        "extras": "train",
        "torch_index": None,  # 기본 PyPI (MPS는 mac wheel에 포함)
        "packages": _BASE_TRAIN_PACKAGES,
    },
    "amd": {
        "label": "AMD ROCm",
        "extras": "train",
        "torch_index": "https://download.pytorch.org/whl/rocm6.0",
        "packages": _BASE_TRAIN_PACKAGES,
    },
    "cpu": {
        "label": "CPU only",
        "extras": "train",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "packages": _BASE_TRAIN_PACKAGES,
    },
}


# ─────────────────────────────────────────
# 진단 결과 모델
# ─────────────────────────────────────────


@dataclass
class EnvDiagnosis:
    os: str
    gpu_vendor: str
    gpu_name: str
    profile: str            # "nvidia" / "apple" / "amd" / "cpu"
    profile_label: str
    cuda_available: bool
    mps_available: bool
    python_version: str
    required_packages: list[str]
    installed_packages: list[str]
    missing_packages: list[str]
    install_cmd: str        # 사용자에게 보여줄 명령
    is_ready: bool          # 누락 없음
    warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────
# 핵심 로직
# ─────────────────────────────────────────


def _detect_profile(caps: HWCaps) -> str:
    """HW 캡스에서 프로파일 결정."""
    for g in caps.gpus:
        if g.vendor == "nvidia":
            return "nvidia"
        if g.vendor == "apple":
            return "apple"
        if g.vendor == "amd":
            return "amd"
    return "cpu"


def _check_imports(packages: dict[str, str]) -> tuple[list[str], list[str]]:
    """import 가능한 패키지 / 누락 패키지 분리.

    `__import__` 는 모듈을 실제 로드하여 module-level 부수효과 (예: bitsandbytes 의
    CUDA 초기화) 가 발생. non-CUDA 호스트에서 RuntimeError/AttributeError 가
    ImportError 가 아닌 형태로 새어나가 진단 자체가 크래시. find_spec 으로 변경하여
    설치 여부만 메타데이터로 확인.
    """
    import importlib.util

    installed, missing = [], []
    for import_name, pip_name in packages.items():
        mod_name = import_name.replace("-", "_")
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            installed.append(pip_name)
        else:
            missing.append(pip_name)
    return installed, missing


def _make_install_cmd_dyn(profile: str, torch_index: str | None) -> str:
    """프로필 + 동적으로 결정된 torch_index로 설치 명령 생성."""
    p = PROFILES[profile]
    parts = ["uv", "pip", "install", f"'.[{p['extras']}]'"]
    if torch_index:
        parts.extend(["--extra-index-url", torch_index])
        parts.extend(["--index-strategy", "unsafe-best-match"])
        # nightly 인덱스면 --prerelease=allow 필요
        if "nightly" in torch_index:
            parts.append("--prerelease=allow")
    return " ".join(parts)


def _make_install_cmd(profile: str) -> str:
    """legacy — 정적 profile torch_index 사용."""
    return _make_install_cmd_dyn(profile, PROFILES[profile].get("torch_index"))


def _resolve_torch_index(profile: str, caps) -> str | None:
    """PyTorch wheel index 동적 결정.

    aarch64 + NVIDIA (Grace+Blackwell/Jetson) → SM 10.x 포함된 nightly cu128 권장.
    aarch64 + NVIDIA + 옛 GPU → nightly cu126.
    그 외 NVIDIA → 표준 cu121.

    NOTE: 환경 변수 WFP_TORCH_INDEX로 사용자 override 가능.
    """
    import os
    if override := os.environ.get("WFP_TORCH_INDEX"):
        return override
    if profile != "nvidia":
        return PROFILES[profile]["torch_index"]
    import platform
    if platform.machine().lower() in ("aarch64", "arm64"):
        # Blackwell (SM 10.x): cu128 nightly. 그 외 aarch64는 cu126.
        # GPU 이름 휴리스틱 — GB10, B100, B200 등 Blackwell 키워드 매칭.
        gpu_names = " ".join(g.name for g in (caps.gpus or [])).upper()
        if any(k in gpu_names for k in ("GB10", "B100", "B200", "BLACKWELL")):
            return "https://download.pytorch.org/whl/nightly/cu128"
        return "https://download.pytorch.org/whl/nightly/cu126"
    return PROFILES["nvidia"]["torch_index"]


def diagnose() -> EnvDiagnosis:
    """현재 환경 진단 + 권장 설치 명령 반환."""
    # HW 프로빙 (실행 환경의 OS·GPU 자동 감지)
    from wfp_worker.stage2_hw_probe import HWProber

    caps = HWProber().probe()
    profile = _detect_profile(caps)
    cfg = dict(PROFILES[profile])
    # aarch64 NVIDIA → nightly cu126 동적 교체
    cfg["torch_index"] = _resolve_torch_index(profile, caps)

    installed, missing = _check_imports(cfg["packages"])
    install_cmd = _make_install_cmd_dyn(profile, cfg["torch_index"])

    # GPU 이름
    gpu_name = caps.gpus[0].name if caps.gpus else "—"
    gpu_vendor = caps.gpus[0].vendor if caps.gpus else "none"

    # 경고
    warnings: list[str] = []
    if profile == "cpu":
        warnings.append("CPU 학습은 매우 느립니다. GPU 또는 클라우드 사용을 권장합니다.")
    if profile == "nvidia" and not caps.cuda_available:
        warnings.append("NVIDIA GPU 감지되었으나 CUDA 비활성 — 드라이버 또는 torch CUDA 빌드 확인 필요.")
    if profile == "apple" and not caps.mps_available:
        # torch 미설치인지 vs 설치됐는데 MPS 불가인지 구분
        torch_installed = "torch" in installed
        if not torch_installed:
            warnings.append(
                "Apple Silicon 감지 — MPS 사용을 위해 torch 학습 패키지 설치 필요. "
                "uv pip install -e '.[train]' --python .venv/bin/python"
            )
        else:
            warnings.append(
                "Apple Silicon 감지 + torch 설치됨 — MPS 비활성. "
                "macOS 12.3+ 필요. torch가 CPU-only wheel일 경우 mac 전용 wheel로 재설치: "
                "uv pip install -U torch --python .venv/bin/python"
            )

    return EnvDiagnosis(
        os=caps.os,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        profile=profile,
        profile_label=cfg["label"],
        cuda_available=caps.cuda_available,
        mps_available=caps.mps_available,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        required_packages=list(cfg["packages"].values()),
        installed_packages=installed,
        missing_packages=missing,
        install_cmd=install_cmd,
        is_ready=len(missing) == 0,
        warning=" / ".join(warnings),
    )


# ─────────────────────────────────────────
# 설치 실행
# ─────────────────────────────────────────


def install_missing(diagnosis: EnvDiagnosis | None = None) -> dict:
    """
    누락된 패키지 자동 설치.

    uv가 있으면 uv pip install, 없으면 현재 Python의 pip으로 fallback.
    완료 후 상태 dict 반환.
    """
    import shutil
    from pathlib import Path

    diag = diagnosis or diagnose()

    if diag.is_ready:
        return {"status": "already_installed", "missing": []}

    cfg = PROFILES[diag.profile]
    # 동적 torch index (aarch64 + nvidia → nightly cu126)
    from wfp_worker.stage2_hw_probe import HWProber
    caps = HWProber().probe()
    torch_index = _resolve_torch_index(diag.profile, caps)

    # uv 탐색 — PATH → 현재 Python venv bin → 없으면 pip fallback
    uv_bin = (
        shutil.which("uv")
        or shutil.which("uv", path=str(Path(sys.executable).parent))
    )

    if uv_bin:
        cmd: list[str] = [uv_bin, "pip", "install", f".[{cfg['extras']}]"]
        if torch_index:
            cmd.extend([
                "--extra-index-url", torch_index,
                "--index-strategy", "unsafe-best-match",
            ])
            if "nightly" in torch_index:
                cmd.append("--prerelease=allow")
    else:
        # pip fallback (--index-strategy 미지원)
        cmd = [sys.executable, "-m", "pip", "install", f".[{cfg['extras']}]"]
        if torch_index:
            cmd.extend(["--extra-index-url", torch_index])
            if "nightly" in torch_index:
                cmd.append("--pre")

    # 프로젝트 루트 자동 감지: env_check.py → wfp_worker/ → src/ → ml-worker/
    _auto_cwd = str(Path(__file__).resolve().parent.parent.parent)
    cwd = os.environ.get("WFP_PROJECT_DIR") or _auto_cwd

    log.info("환경 패키지 자동 설치 시작",
             extra={"extra": {"cmd": " ".join(cmd), "cwd": cwd, "missing": diag.missing_packages}})

    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        return {
            "status": "uv_not_found",
            "detail": f"uv/pip 실행 불가. 수동 설치: cd {cwd} && {diag.install_cmd}",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": "설치 10분 초과 — 수동 실행 권장."}

    payload = {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }

    if payload["status"] == "ok":
        # 재진단
        post = diagnose()
        payload["still_missing"] = post.missing_packages
        payload["restart_required"] = True  # FastAPI는 import된 모듈 reload 안 됨

    return payload


# ─────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────


def _print_diagnosis(d: EnvDiagnosis) -> None:
    print(f"\n환경 진단 (Environment Diagnosis)")
    print(f"  OS         : {d.os}")
    print(f"  Profile    : {d.profile_label} ({d.profile})")
    print(f"  GPU        : {d.gpu_vendor} — {d.gpu_name}")
    print(f"  Python     : {d.python_version}")
    print(f"  CUDA       : {d.cuda_available}")
    print(f"  MPS        : {d.mps_available}")
    print()
    print(f"  Installed  : {len(d.installed_packages)} / {len(d.required_packages)}")
    if d.missing_packages:
        print(f"  Missing    : {', '.join(d.missing_packages)}")
        print(f"  Install    : {d.install_cmd}")
    else:
        print(f"  Status     : ✓ ready")
    if d.warning:
        print(f"  Warning    : {d.warning}")
    print()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="WFP 환경 진단 + 자동 설치")
    p.add_argument("--install", action="store_true", help="누락 패키지 자동 설치")
    p.add_argument("--json", action="store_true", help="JSON 출력")
    p.add_argument("--strict", action="store_true",
                   help="진단 실패(누락 패키지)를 exit code로 표시 (CI/스크립트용). 기본은 정보 출력만 후 exit 0.")
    args = p.parse_args()

    diag = diagnose()

    if args.json:
        import json
        print(json.dumps(diag.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_diagnosis(diag)

    if args.install and not diag.is_ready:
        print("→ 자동 설치 시작...")
        result = install_missing(diag)
        print(f"결과: {result['status']}")
        if result.get("stderr"):
            print(f"stderr: {result['stderr'][-500:]}")
        return 0 if result["status"] == "ok" else 1

    # 단순 진단은 항상 exit 0 (정보 출력만). --strict 일 때만 누락 시 exit 2.
    if args.strict and not diag.is_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
