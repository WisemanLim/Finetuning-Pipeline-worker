"""
Stage 9 보조 — 파인튜닝 완료 모델 on-demand 포맷 변환.

지원 포맷 (ArtifactKind):
  merged  — 베이스 + LoRA 어댑터 병합 (HF safetensors). 다른 변환의 선행 조건.
  gguf    — llama.cpp convert_hf_to_gguf.py (+ llama-quantize). WFP_LLAMA_CPP_DIR.
  mlx     — mlx_lm.convert (Apple Silicon)
  pytorch — safetensors → torch.save(pytorch_model.bin)
  onnx    — optimum-cli export onnx

설계 원칙:
  - merged 외 포맷은 전부 merged 디렉토리(전체 가중치)에서 출발.
    LoRA 어댑터 단독으로는 GGUF/MLX/ONNX 변환 불가.
  - 도구 미설치 시 ToolingMissingError → 호출부가 409 + 설치 힌트 반환.
  - 동기 함수 — 호출부에서 asyncio.to_thread 로 백그라운드 실행.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import Artifact, ArtifactKind, JobSpec, TrainStrategy

log = get_logger(__name__)

# 변환 서브프로세스 타임아웃 (초). 대형 모델 대비 기본 30분.
_CONVERT_TIMEOUT = int(os.environ.get("WFP_CONVERT_TIMEOUT", "1800"))

# GGUF 허용 양자화 타입. convert 단계는 항상 f16, 이후 llama-quantize.
_GGUF_QUANTS = ("f16", "f32", "q8_0", "Q4_K_M", "Q5_K_M", "Q6_K")
_DEFAULT_GGUF_QUANT = "Q4_K_M"


class ToolingMissingError(Exception):
    """변환 도구(스크립트/패키지/바이너리) 미설치."""

    def __init__(self, fmt: str, hint: str) -> None:
        self.fmt = fmt
        self.hint = hint
        super().__init__(f"{fmt} 변환 도구 없음: {hint}")


class ConvertFailedError(Exception):
    """변환 실행 실패 (서브프로세스/런타임 오류)."""


# ─────────────────────────────────────────
# 도구 가용성
# ─────────────────────────────────────────


def _llama_convert_script() -> str | None:
    """llama.cpp convert_hf_to_gguf.py 경로. WFP_LLAMA_CPP_DIR 우선, PATH 폴백."""
    root = os.environ.get("WFP_LLAMA_CPP_DIR", "").strip()
    if root:
        cand = Path(root) / "convert_hf_to_gguf.py"
        if cand.exists():
            return str(cand)
    return shutil.which("convert_hf_to_gguf.py") or shutil.which("convert.py")


def _llama_quantize_bin() -> str | None:
    """llama-quantize (신) / quantize (구) 바이너리 경로."""
    root = os.environ.get("WFP_LLAMA_CPP_DIR", "").strip()
    if root:
        for name in ("llama-quantize", "quantize", "build/bin/llama-quantize"):
            cand = Path(root) / name
            if cand.exists():
                return str(cand)
    return shutil.which("llama-quantize") or shutil.which("quantize")


def _has_pkg(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def tooling_status() -> dict[str, dict]:
    """포맷별 변환 가능 여부 + 설치 힌트.

    Frontend 가 버튼 활성화/비활성화 + 안내에 사용.
    """
    torch_ok = _has_pkg("torch")
    merged_ok = torch_ok and _has_pkg("transformers") and _has_pkg("peft")
    return {
        "merged": {
            "available": merged_ok,
            "hint": "" if merged_ok else "torch / transformers / peft 필요 (make install-deps)",
        },
        "gguf": {
            "available": bool(_llama_convert_script()),
            "hint": "" if _llama_convert_script() else
                    "llama.cpp 필요 — make install-convert-tools (WFP_LLAMA_CPP_DIR)",
        },
        "mlx": {
            "available": _has_pkg("mlx_lm"),
            "hint": "" if _has_pkg("mlx_lm") else
                    "mlx-lm 필요 (Apple Silicon) — make install-convert-tools",
        },
        "pytorch": {
            "available": torch_ok and _has_pkg("safetensors"),
            "hint": "" if (torch_ok and _has_pkg("safetensors")) else
                    "torch / safetensors 필요 (make install-deps)",
        },
        "onnx": {
            "available": _has_pkg("optimum") and _has_pkg("onnx"),
            "hint": "" if (_has_pkg("optimum") and _has_pkg("onnx")) else
                    "optimum[exporters] + onnx 필요 — make install-convert-tools",
        },
    }


# ─────────────────────────────────────────
# merged (선행 조건)
# ─────────────────────────────────────────


def ensure_merged(spec: JobSpec, out_dir: Path) -> Path:
    """베이스 + 어댑터 병합 디렉토리 보장. 이미 있으면 재사용.

    반환: merged 디렉토리 경로 (config.json + *.safetensors + tokenizer).
    """
    merged_dir = out_dir / "merged"
    if (merged_dir / "config.json").exists():
        return merged_dir

    ckpt = spec.checkpoint_dir
    if not ckpt or not Path(ckpt).exists():
        raise ConvertFailedError("checkpoint_dir 없음 — Stage 7(학습)을 먼저 완료하세요.")

    # Full Fine-tuning: 체크포인트 자체가 전체 가중치 → 어댑터 병합 불필요, 그대로 복사.
    strategy = spec.train_config.strategy if spec.train_config else None
    if strategy == TrainStrategy.FULL_FT:
        if (Path(ckpt) / "config.json").exists():
            merged_dir.mkdir(parents=True, exist_ok=True)
            log.info("convert_merge_fullft_copy", extra={"extra": {"ckpt": ckpt}})
            shutil.copytree(ckpt, merged_dir, dirs_exist_ok=True)
            return merged_dir
        log.warning("fullft_ckpt_no_config",
                    extra={"extra": {"ckpt": ckpt}})  # fall through to peft merge attempt

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ToolingMissingError(
            "merged", "torch / transformers / peft 필요 (make install-deps)"
        ) from e

    merged_dir.mkdir(parents=True, exist_ok=True)
    base_ref = spec.model_local_path or spec.model_hf_repo
    log.info("convert_merge_start", extra={"extra": {"base": base_ref, "ckpt": ckpt}})

    base_model = AutoModelForCausalLM.from_pretrained(
        base_ref, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(ckpt))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))
    return merged_dir


# ─────────────────────────────────────────
# 포맷별 변환
# ─────────────────────────────────────────


def _model_name(spec: JobSpec) -> str:
    return (spec.model_hf_repo or "model").split("/")[-1]


def _convert_gguf(spec: JobSpec, merged_dir: Path, out_dir: Path, quant: str) -> Path:
    script = _llama_convert_script()
    if not script:
        raise ToolingMissingError(
            "gguf", "llama.cpp 필요 — make install-convert-tools (WFP_LLAMA_CPP_DIR)"
        )
    if quant not in _GGUF_QUANTS:
        quant = _DEFAULT_GGUF_QUANT

    name = _model_name(spec)
    f16_path = out_dir / f"{name}.f16.gguf"
    _run(
        ["python", script, str(merged_dir), "--outtype", "f16", "--outfile", str(f16_path)],
        "gguf",
    )

    if quant in ("f16", "f32"):
        final = out_dir / f"{name}.{quant}.gguf"
        f16_path.replace(final)
        return final

    qbin = _llama_quantize_bin()
    if not qbin:
        # 양자화 바이너리 없음 — f16 결과라도 반환 (다운로드 가능).
        log.warning("llama-quantize 없음 — f16 GGUF 로 폴백", extra={"extra": {"quant": quant}})
        final = out_dir / f"{name}.f16.gguf"
        return final if final.exists() else f16_path

    final = out_dir / f"{name}.{quant}.gguf"
    _run([qbin, str(f16_path), str(final), quant], "gguf")
    f16_path.unlink(missing_ok=True)
    return final


def _convert_mlx(spec: JobSpec, merged_dir: Path, out_dir: Path, quant: str) -> Path:
    if not _has_pkg("mlx_lm"):
        raise ToolingMissingError("mlx", "mlx-lm 필요 (Apple Silicon) — make install-convert-tools")
    mlx_dir = out_dir / "mlx"
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir, ignore_errors=True)
    cmd = ["python", "-m", "mlx_lm.convert",
           "--hf-path", str(merged_dir), "--mlx-path", str(mlx_dir)]
    if quant and quant.lower() in ("int4", "q4", "4bit", "quantize"):
        cmd.append("-q")
    _run(cmd, "mlx")
    return mlx_dir


def _convert_pytorch(spec: JobSpec, merged_dir: Path, out_dir: Path) -> Path:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as e:
        raise ToolingMissingError("pytorch", "torch / safetensors 필요 (make install-deps)") from e

    pt_dir = out_dir / "pytorch"
    pt_dir.mkdir(parents=True, exist_ok=True)
    # config / tokenizer 도 함께 복사 — 로딩에 필요.
    for f in merged_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, pt_dir / f.name)

    state: dict = {}
    shards = sorted(merged_dir.glob("*.safetensors"))
    if not shards:
        raise ConvertFailedError("merged 디렉토리에 *.safetensors 없음")
    for shard in shards:
        state.update(load_file(str(shard)))
    torch.save(state, str(pt_dir / "pytorch_model.bin"))
    return pt_dir


def _convert_onnx(spec: JobSpec, merged_dir: Path, out_dir: Path) -> Path:
    if not (_has_pkg("optimum") and _has_pkg("onnx")):
        raise ToolingMissingError(
            "onnx", "optimum[exporters] + onnx 필요 — make install-convert-tools"
        )
    onnx_dir = out_dir / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir, ignore_errors=True)
    _run(
        ["optimum-cli", "export", "onnx", "--model", str(merged_dir), str(onnx_dir)],
        "onnx",
    )
    return onnx_dir


# ─────────────────────────────────────────
# 디스패처
# ─────────────────────────────────────────


def convert(
    spec: JobSpec,
    kind: ArtifactKind,
    deploy_root: Path,
    *,
    quant: str = _DEFAULT_GGUF_QUANT,
) -> Artifact:
    """포맷 변환 실행 → Artifact 반환. 호출부에서 spec.artifacts 에 추가.

    Raises
    ------
    ToolingMissingError  도구 미설치 (409)
    ConvertFailedError   변환 실패 (500)
    """
    out_dir = deploy_root / str(spec.job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == ArtifactKind.MERGED:
        path = ensure_merged(spec, out_dir)
    else:
        merged_dir = ensure_merged(spec, out_dir)
        if kind == ArtifactKind.GGUF:
            path = _convert_gguf(spec, merged_dir, out_dir, quant)
        elif kind == ArtifactKind.MLX:
            path = _convert_mlx(spec, merged_dir, out_dir, quant)
        elif kind == ArtifactKind.PYTORCH:
            path = _convert_pytorch(spec, merged_dir, out_dir)
        elif kind == ArtifactKind.ONNX:
            path = _convert_onnx(spec, merged_dir, out_dir)
        else:
            raise ConvertFailedError(f"변환 불가 kind: {kind.value}")

    if path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        sha = ""
    else:
        size = path.stat().st_size if path.exists() else 0
        sha = _sha256(path)

    log.info("convert_done",
             extra={"job_id": str(spec.job_id),
                    "extra": {"kind": kind.value, "path": str(path), "bytes": size}})
    return Artifact(kind=kind, uri=str(path), size_bytes=size, sha256=sha)


def _run(cmd: list[str], fmt: str) -> None:
    """서브프로세스 실행 + 오류를 ConvertFailedError 로 정규화."""
    try:
        subprocess.run(cmd, check=True, timeout=_CONVERT_TIMEOUT)
    except subprocess.CalledProcessError as e:
        raise ConvertFailedError(f"{fmt} 변환 실패 (exit {e.returncode})") from e
    except subprocess.TimeoutExpired as e:
        raise ConvertFailedError(f"{fmt} 변환 타임아웃 ({_CONVERT_TIMEOUT}s)") from e
    except FileNotFoundError as e:
        raise ConvertFailedError(f"{fmt} 변환 명령 없음: {cmd[0]}") from e


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
