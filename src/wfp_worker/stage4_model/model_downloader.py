"""
Stage 4 — 모델 다운로더.

HuggingFace Hub에서 모델을 로컬 캐시 디렉토리로 다운로드한다.
캐시 경로: ~/.wfp/models/<org>/<model>/<revision>

다운로드 전 디스크 여유 공간을 확인하고,
이미 존재하면 재다운로드 없이 캐시 경로를 반환한다.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobSpec, JobState

log = get_logger(__name__)

# 기본 캐시 루트
_DEFAULT_CACHE_ROOT = Path.home() / ".wfp" / "models"

# 최소 필요 디스크 여유 공간 (모델 크기 추정치 대비 배수)
_DISK_SAFETY_MULTIPLIER = 1.5


class ModelDownloader:
    """
    Stage 4: 모델 다운로드 및 캐시 관리.

    책임:
    - 디스크 공간 사전 확인
    - huggingface_hub.snapshot_download 호출
    - 캐시 디렉토리 경로 반환 및 JobSpec 갱신
    - HF 토큰 안전 로드 (환경 변수 우선)
    """

    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root or _DEFAULT_CACHE_ROOT
        self._cache_root.mkdir(parents=True, exist_ok=True)

    def download(self, spec: JobSpec, revision: str = "main") -> str:
        """
        모델 다운로드 또는 캐시 재사용.

        Parameters
        ----------
        spec:
            model_hf_repo가 설정된 JobSpec.
        revision:
            브랜치/태그/커밋 SHA. 기본값 "main".

        Returns
        -------
        str
            모델이 저장된 로컬 디렉토리 경로.

        Raises
        ------
        WFPError(WFP-0404)
            디스크 공간 부족.
        WFPError(WFP-0401)
            HuggingFace 다운로드 실패.
        WFPError(WFP-0405)
            인증이 필요한 모델인데 토큰 없음.
        """
        repo_id = spec.model_hf_repo
        if not repo_id:
            raise WFPError(
                code=WFPErrorCode.MODEL_NOT_FOUND,
                message="JobSpec에 model_hf_repo가 설정되지 않았습니다.",
                stage=4,
            )

        log.info(
            "Stage 4: 모델 다운로드 시작",
            extra={"stage": 4, "job_id": str(spec.job_id), "extra": {"repo": repo_id, "revision": revision}},
        )

        # 캐시 경로 계산
        safe_name = repo_id.replace("/", "__")
        local_dir = self._cache_root / safe_name / revision

        # 이미 다운로드된 경우 재사용
        if self._is_cached(local_dir):
            log.info(
                "Stage 4: 캐시 재사용",
                extra={"stage": 4, "extra": {"path": str(local_dir)}},
            )
            spec.model_local_path = str(local_dir)
            spec.model_revision = revision
            spec.update_stage(stage=4, state=JobState.RUNNING)
            return str(local_dir)

        # 디스크 공간 확인 (모델 크기를 params_b × 2GB로 추정)
        self._check_disk_space(spec, local_dir)

        # HF 토큰 로드
        token = self._load_token()

        # 다운로드 실행
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(local_dir),
                token=token,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.HF_DOWNLOAD_FAILED,
                message="huggingface_hub 미설치. pip install huggingface-hub",
                stage=4,
            ) from None
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "authentication" in err_str:
                raise WFPError(
                    code=WFPErrorCode.HF_TOKEN_MISSING,
                    message=f"'{repo_id}' 접근에 HF 토큰이 필요합니다.",
                    stage=4,
                    detail="HF_TOKEN 환경 변수를 설정하세요.",
                ) from e
            raise WFPError(
                code=WFPErrorCode.HF_DOWNLOAD_FAILED,
                message=f"모델 다운로드 실패: {repo_id}",
                stage=4,
                detail=str(e),
            ) from e

        spec.model_local_path = path
        spec.model_revision = revision
        spec.update_stage(stage=4, state=JobState.RUNNING)

        log.info(
            "Stage 4: 다운로드 완료",
            extra={"stage": 4, "job_id": str(spec.job_id), "extra": {"path": path}},
        )

        return path

    # ─────────────────────────────────────────
    # 내부 유틸
    # ─────────────────────────────────────────

    @staticmethod
    def _is_cached(local_dir: Path) -> bool:
        """config.json이 존재하면 이미 다운로드 완료로 판단."""
        return (local_dir / "config.json").exists()

    def _check_disk_space(self, spec: JobSpec, local_dir: Path) -> None:
        """모델 크기 추정치와 디스크 여유 공간 비교."""
        # 카탈로그에서 params_b 조회 시도
        from .model_catalog import _ALL_MODELS

        matched = next((m for m in _ALL_MODELS if m.hf_repo == spec.model_hf_repo), None)
        estimated_gb = (matched.params_b * 2.0) if matched else 14.0  # 기본값 14GB

        required_gb = estimated_gb * _DISK_SAFETY_MULTIPLIER
        free_gb = psutil.disk_usage(str(self._cache_root)).free / (1024 ** 3)

        if free_gb < required_gb:
            raise WFPError(
                code=WFPErrorCode.INSUFFICIENT_DISK,
                message=f"디스크 공간 부족. 필요: {required_gb:.1f}GB, 여유: {free_gb:.1f}GB",
                stage=4,
            )

    @staticmethod
    def _load_token() -> str | None:
        """환경 변수 → ~/.cache/huggingface/token 순으로 토큰 로드."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token

        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            return token_path.read_text().strip() or None

        return None
