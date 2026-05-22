"""
Stage 9 — 패키징 및 배포 (Packager).

PRD §4.9: 학습 완료 모델을 다양한 포맷으로 패키징.

아티팩트 종류:
  1. LoRA 어댑터 (adapter_model.safetensors) — 항상 생성
  2. 머지 weights (optional) — 베이스 + 어댑터 병합
  3. GGUF 변환본 (optional) — llama.cpp ctransformers 사용
  4. model_card.md (자동 생성) — 항상 생성
  5. ZIP 패키지 (optional)

다운로드 URL은 object storage (MinIO/S3) 업로드 후 생성.
데스크탑 모드에서는 로컬 경로를 그대로 반환.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import Artifact, ArtifactKind, JobSpec, JobState

log = get_logger(__name__)

# 기본 배포 루트
_DEFAULT_DEPLOY_ROOT = Path.home() / ".wfp" / "artifacts"


class Packager:
    """
    Stage 9: 아티팩트 패키징.

    책임:
    - 어댑터 파일 검증 및 SHA256 계산
    - model_card.md 자동 생성
    - 머지 weights 생성 (optional)
    - GGUF 변환 (optional)
    - ZIP 아카이브 생성 (optional)
    - Artifact 목록 생성 및 JobSpec.artifacts 갱신
    """

    def __init__(self, deploy_root: Path | None = None) -> None:
        self._deploy_root = deploy_root or _DEFAULT_DEPLOY_ROOT

    def package(
        self,
        spec: JobSpec,
        create_merged: bool = False,
        create_gguf: bool = False,
        create_zip: bool = True,
    ) -> list[Artifact]:
        """
        아티팩트 패키징 실행.

        Parameters
        ----------
        spec:
            Stage 8까지 채워진 JobSpec.
        create_merged:
            베이스 모델 + 어댑터를 병합한 full weights 생성 여부.
        create_gguf:
            GGUF 변환 파일 생성 여부 (llama.cpp 필요).
        create_zip:
            모든 아티팩트를 ZIP으로 묶어 다운로드 제공.

        Returns
        -------
        list[Artifact]
            생성된 아티팩트 목록.
        """
        checkpoint = spec.checkpoint_dir
        if not checkpoint:
            raise WFPError(
                code=WFPErrorCode.ARTIFACT_UPLOAD_FAILED,
                message="checkpoint_dir 없음. Stage 7을 먼저 실행하세요.",
                stage=9,
            )

        out_dir = self._deploy_root / str(spec.job_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[Artifact] = []
        ckpt_path = Path(checkpoint)

        log.info("Stage 9: 패키징 시작", extra={"stage": 9, "job_id": str(spec.job_id)})

        # 1. 어댑터 파일
        adapter_artifact = self._package_adapter(ckpt_path, out_dir)
        if adapter_artifact:
            artifacts.append(adapter_artifact)

        # 2. model_card.md
        card_artifact = self._generate_model_card(spec, out_dir)
        artifacts.append(card_artifact)

        # 3. 머지 weights (optional)
        if create_merged:
            try:
                merged_artifact = self._merge_weights(spec, ckpt_path, out_dir)
                if merged_artifact:
                    artifacts.append(merged_artifact)
            except Exception as e:
                log.warning("머지 weights 생성 실패", extra={"extra": {"err": str(e)}})

        # 4. GGUF 변환 (optional)
        if create_gguf:
            try:
                gguf_artifact = self._convert_gguf(spec, out_dir)
                if gguf_artifact:
                    artifacts.append(gguf_artifact)
            except Exception as e:
                log.warning("GGUF 변환 실패", extra={"extra": {"err": str(e)}})

        # 5. ZIP 패키지 (optional)
        if create_zip and artifacts:
            zip_artifact = self._create_zip(artifacts, out_dir)
            artifacts.append(zip_artifact)

        # JobSpec 갱신
        spec.artifacts = artifacts
        spec.update_stage(stage=9, state=JobState.DONE)

        log.info(
            "Stage 9: 패키징 완료",
            extra={
                "stage": 9,
                "job_id": str(spec.job_id),
                "extra": {
                    "artifact_count": len(artifacts),
                    "kinds": [a.kind.value for a in artifacts],
                },
            },
        )

        return artifacts

    # ─────────────────────────────────────────
    # 아티팩트 생성 메서드
    # ─────────────────────────────────────────

    @staticmethod
    def _package_adapter(ckpt_path: Path, out_dir: Path) -> Artifact | None:
        """어댑터 파일 복사 및 Artifact 생성."""
        adapter_file = ckpt_path / "adapter_model.safetensors"
        alt_files = [
            ckpt_path / "adapter_model.bin",
            ckpt_path / "pytorch_model.bin",
        ]

        # safetensors 우선, 없으면 대안
        source = adapter_file if adapter_file.exists() else next(
            (f for f in alt_files if f.exists()), None
        )

        if not source:
            log.warning("어댑터 파일 없음", extra={"extra": {"ckpt": str(ckpt_path)}})
            return None

        dest = out_dir / source.name
        if source != dest:
            shutil.copy2(source, dest)

        # adapter_config.json도 복사
        config_file = ckpt_path / "adapter_config.json"
        if config_file.exists():
            shutil.copy2(config_file, out_dir / "adapter_config.json")

        return Artifact(
            kind=ArtifactKind.ADAPTER,
            uri=str(dest),
            size_bytes=dest.stat().st_size,
            sha256=_sha256(dest),
        )

    def _generate_model_card(self, spec: JobSpec, out_dir: Path) -> Artifact:
        """model_card.md 자동 생성."""
        card_path = out_dir / "model_card.md"

        eval_info = ""
        if spec.eval_result:
            r = spec.eval_result
            eval_info = f"""
## 평가 결과

| 메트릭 | 값 |
|--------|-----|
| Perplexity | {r.perplexity or 'N/A'} |
| ROUGE-1 | {r.rouge1 or 'N/A'} |
| ROUGE-L | {r.rougeL or 'N/A'} |
| F1 | {r.f1 or 'N/A'} |
| 품질 기준 통과 | {'✅ Yes' if r.passed else '❌ No'} |
"""

        content = f"""# {spec.model_hf_repo} — WFP Fine-tuned

> 자동 생성: Wise Fine-tuning Pipeline v0.1 | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

## 모델 정보

- **베이스 모델**: [{spec.model_hf_repo}](https://huggingface.co/{spec.model_hf_repo})
- **KSIC 섹션**: {spec.ksic_section} / 중분류: {spec.ksic_division or 'N/A'}
- **태스크 타입**: {spec.task_type.value}
- **학습 언어**: {spec.language}
- **Job ID**: `{spec.job_id}`

## 학습 설정

```json
{json.dumps(spec.train_config.model_dump() if spec.train_config else {}, indent=2, ensure_ascii=False)}
```

## 데이터셋

- 학습: {spec.num_train}건
- 검증: {spec.num_val}건
- 테스트: {spec.num_test}건
- 소스 모드: {spec.data_source_mode.value}
{eval_info}
## 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{spec.model_hf_repo}")
model = PeftModel.from_pretrained(base, "<adapter_path>")
tokenizer = AutoTokenizer.from_pretrained("{spec.model_hf_repo}")
```

## 라이선스

베이스 모델의 라이선스를 따릅니다. 사용 전 확인하세요.
"""

        card_path.write_text(content, encoding="utf-8")

        return Artifact(
            kind=ArtifactKind.MODEL_CARD,
            uri=str(card_path),
            size_bytes=card_path.stat().st_size,
            sha256=_sha256(card_path),
        )

    @staticmethod
    def _merge_weights(spec: JobSpec, ckpt_path: Path, out_dir: Path) -> Artifact | None:
        """베이스 모델 + 어댑터 병합 (PEFT merge_and_unload)."""
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.MISSING_DEPENDENCY,
                message="머지에 torch, transformers, peft 필요",
                stage=9,
            ) from None

        merged_dir = out_dir / "merged"
        merged_dir.mkdir(exist_ok=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            spec.model_local_path or spec.model_hf_repo,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_local_path or spec.model_hf_repo,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(str(merged_dir))

        # 크기 합산
        total_size = sum(f.stat().st_size for f in merged_dir.rglob("*") if f.is_file())

        return Artifact(
            kind=ArtifactKind.MERGED,
            uri=str(merged_dir),
            size_bytes=total_size,
        )

    @staticmethod
    def _convert_gguf(spec: JobSpec, out_dir: Path) -> Artifact | None:
        """
        GGUF 포맷 변환 (llama.cpp convert 스크립트 사용).
        llama.cpp가 PATH에 없으면 생략.
        """
        import shutil
        import subprocess

        convert_script = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert.py")
        if not convert_script:
            log.warning("llama.cpp convert 스크립트 없음 — GGUF 생략")
            return None

        gguf_path = out_dir / f"{spec.model_hf_repo.split('/')[-1]}.Q4_K_M.gguf"
        try:
            subprocess.run(
                [
                    "python", convert_script,
                    spec.checkpoint_dir,
                    "--outtype", "q4_k_m",
                    "--outfile", str(gguf_path),
                ],
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise WFPError(
                code=WFPErrorCode.GGUF_CONVERT_FAILED,
                message="GGUF 변환 실패",
                stage=9,
                detail=str(e),
            ) from e

        return Artifact(
            kind=ArtifactKind.GGUF,
            uri=str(gguf_path),
            size_bytes=gguf_path.stat().st_size if gguf_path.exists() else 0,
            sha256=_sha256(gguf_path) if gguf_path.exists() else "",
        )

    @staticmethod
    def _create_zip(artifacts: list[Artifact], out_dir: Path) -> Artifact:
        """아티팩트를 ZIP으로 묶기."""
        zip_path = out_dir / "package.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for artifact in artifacts:
                art_path = Path(artifact.uri)
                if art_path.is_dir():
                    for file in art_path.rglob("*"):
                        if file.is_file():
                            zf.write(file, file.relative_to(art_path.parent))
                elif art_path.is_file():
                    zf.write(art_path, art_path.name)

        return Artifact(
            kind=ArtifactKind.ZIP,
            uri=str(zip_path),
            size_bytes=zip_path.stat().st_size,
            sha256=_sha256(zip_path),
        )


# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────

def _sha256(path: Path) -> str:
    """파일 SHA256 계산 (대용량 파일 스트리밍)."""
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
