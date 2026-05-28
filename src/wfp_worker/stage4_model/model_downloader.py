"""
Stage 4 — 모델 다운로더.

HuggingFace Hub에서 모델을 로컬 캐시 디렉토리로 다운로드한다.
캐시 경로: ~/.wfp/models/<org>/<model>/<revision>

다운로드 전 디스크 여유 공간을 확인하고,
이미 존재하면 재다운로드 없이 캐시 경로를 반환한다.

진행률:
  Stage 7/8 과 동일한 progress_cb(job_id, stage, msg, extra) 콜백 지원.
  snapshot_download 의 `tqdm_class` 파라미터에 closure-bound 클래스를 주입하여
  파일 단위 집계 진행도를 WS 이벤트로 발행. contextvars 미전파 worker 스레드에서도
  동작하도록 콜백을 클래스 변수로 캡처한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import psutil

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobSpec, JobState

log = get_logger(__name__)

# Stage 7/8 trainer/evaluator 와 동일 시그니처.
ProgressCb = Callable[[str, int, str, dict[str, Any]], None]


def _build_emitting_tqdm_class(job_id: str, cb: ProgressCb) -> type:
    """snapshot_download(tqdm_class=...) 에 주입할 tqdm 서브클래스 생성.

    snapshot_download 내부는 ThreadPoolExecutor 로 병렬 파일 다운로드를 돌리며
    `tqdm_class` 는 "완료된 파일 수" aggregated bar 에 사용된다. contextvar 기반
    tqdm_bridge 는 worker 스레드에서 job_id 를 못 잡으므로, 여기서는 클로저로
    (job_id, cb) 를 캡처해 직접 emit.
    """
    try:
        from tqdm.std import tqdm as _BaseTqdm
    except Exception:
        # tqdm 미설치 — fallback to None (snapshot_download 가 기본 tqdm 사용).
        return None  # type: ignore[return-value]

    class _DownloadTqdm(_BaseTqdm):  # type: ignore[misc, valid-type]
        def update(self, n: float = 1) -> Any:
            r = super().update(n)
            self._wfp_emit()
            return r

        def refresh(self, *args: Any, **kwargs: Any) -> Any:
            r = super().refresh(*args, **kwargs)
            self._wfp_emit()
            return r

        def close(self) -> Any:
            try:
                self._wfp_emit()
            except Exception:
                pass
            return super().close()

        def _wfp_emit(self) -> None:
            try:
                total = getattr(self, "total", None) or 0
                if total <= 0:
                    return
                n = getattr(self, "n", 0) or 0
                desc = getattr(self, "desc", None) or "Stage 4: 모델 다운로드"
                cb(
                    job_id, 4,
                    f"{desc}: {int(n)}/{int(total)}",
                    {
                        "step": int(n),
                        "total_steps": int(total),
                        "tqdm_desc": desc,
                        "stage_progress": (n / total) if total else 0.0,
                    },
                )
            except Exception:
                pass

    return _DownloadTqdm

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

    def __init__(
        self,
        cache_root: Path | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> None:
        self._cache_root = cache_root or _DEFAULT_CACHE_ROOT
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._progress_cb = progress_cb

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

        # HF 토큰 로드 — spec.hf_token (잡 생성 시 user.hf_token_encrypted 스냅샷) 우선,
        # 없으면 환경 변수 / huggingface 캐시 토큰 순.
        token = self._load_token(spec)

        # 진행도 발행 — 시작 마커
        if self._progress_cb is not None:
            try:
                self._progress_cb(
                    str(spec.job_id), 4,
                    f"Stage 4: HF 모델 다운로드 시작 ({repo_id})",
                    {"tqdm_desc": "Stage 4: model download", "step": 0,
                     "total_steps": 0, "stage_progress": 0.0},
                )
            except Exception:
                pass

        # 다운로드 실행
        try:
            from huggingface_hub import snapshot_download

            kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": str(local_dir),
                "token": token,
                "ignore_patterns": ["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            }
            # snapshot_download 의 aggregated 파일 진행도 → WS 이벤트로 발행.
            if self._progress_cb is not None:
                tqdm_cls = _build_emitting_tqdm_class(str(spec.job_id), self._progress_cb)
                if tqdm_cls is not None:
                    kwargs["tqdm_class"] = tqdm_cls
            path = snapshot_download(**kwargs)
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.HF_DOWNLOAD_FAILED,
                message="huggingface_hub 미설치. pip install huggingface-hub",
                stage=4,
            ) from None
        except Exception as e:
            err_str = str(e).lower()
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            # 403 = gated repo (인증 OK, 권한/라이선스 동의 부족) — 관리자 알림 흐름.
            if (
                status_code == 403
                or "403" in err_str
                or "gated" in err_str
                or ("access" in err_str and "denied" in err_str)
            ):
                raise WFPError(
                    code=WFPErrorCode.LICENSE_NOT_ACCEPTED,
                    message=f"'{repo_id}' 는 gated repo — 라이선스 동의 + HF 측 승인 필요.",
                    stage=4,
                    detail=(
                        f"관리자가 https://huggingface.co/{repo_id} 방문 → Access repository "
                        f"클릭 → 라이선스 form 제출 → 승인 후 /admin/jobs/{{job_id}}/resume-download "
                        f"호출. 일부 모델은 자동 승인, 일부는 수동 review."
                    ),
                ) from e
            # 401 = 토큰 자체 없음/무효.
            if status_code == 401 or "401" in err_str or "authentication" in err_str or "unauthorized" in err_str:
                raise WFPError(
                    code=WFPErrorCode.HF_TOKEN_MISSING,
                    message=f"'{repo_id}' 접근에 HF 토큰이 필요합니다.",
                    stage=4,
                    detail="관리자 호스트의 HF_TOKEN 환경 변수 / ~/.cache/huggingface/token 설정 확인.",
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
    def _load_token(spec: JobSpec | None = None) -> str | None:
        """JobSpec.hf_token → 환경 변수 → ~/.cache/huggingface/token 순으로 토큰 로드.

        사용자가 /users/me/hf-token 으로 저장한 토큰은 jobs.create_job 단계에서
        JobSpec.hf_token 에 스냅샷됨. 잡 단위 토큰이 환경 변수보다 우선.
        """
        if spec is not None and spec.hf_token:
            t = spec.hf_token.strip()
            if t:
                return t

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token.strip() or None

        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            return token_path.read_text().strip() or None

        return None
