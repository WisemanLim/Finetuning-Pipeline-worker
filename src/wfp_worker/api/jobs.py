"""
잡 라우터.

기존 server.py의 잡 관련 엔드포인트를 분리 + 사용자별 이력 조회 추가.

엔드포인트:
  POST   /jobs                잡 생성 + 파이프라인 비동기 시작
  GET    /jobs                현재 사용자 잡 목록 (state=... 필터)
  GET    /jobs/{id}           잡 상태 조회
  DELETE /jobs/{id}           잡 취소
  GET    /jobs/{id}/artifacts/{kind}  아티팩트 파일 다운로드 (FileResponse)

WebSocket `/ws/{id}`는 server.py에서 직접 유지 (라우터 분리 복잡도 ↓).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from wfp_worker.auth import get_current_user
from wfp_worker.common.errors import WFPError
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import ArtifactKind, JobSpec, JobState
from wfp_worker.db import AdminNotification, User, get_db

log = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


# 잡 인메모리 스토어 — server.py와 공유 (모듈 임포트 시 동일 인스턴스).
from wfp_worker.runtime import (  # noqa: E402
    _job_reqs,
    _job_tasks,
    _jobs,
    _run_pipeline,
    _ws_clients,
    advance_job,
    finalize_job,
    get_last_event_at,
)

# ─────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────


class JobCreateRequest(BaseModel):
    ksic_section: str
    ksic_division: str = ""
    task_type: str = "chat"
    description: str
    language: str = "ko"
    model_hf_repo: str = ""
    data_source_mode: str = "upload"
    data_source_path: str = ""
    keywords: list[str] = []
    train_override: dict[str, Any] = {}
    # 사용자 동의 시 HW 규칙 매칭 실패에도 fallback plan으로 강제 진행 (PRD §8 예외 처리)
    allow_force_plan: bool = False


class JobListItem(BaseModel):
    job_id: str
    state: str
    current_stage: int
    ksic_section: str
    task_type: str
    model_hf_repo: str
    created_at: str | None
    error_code: str = ""


class JobDetail(JobListItem):
    error_code: str = ""
    error_message: str = ""
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    artifacts: list[dict] = []
    # 재실행(재수정) 위자드 prefill용 spec 노출
    ksic_division: str = ""
    description: str = ""
    language: str = "ko"
    data_source_mode: str = ""
    data_source_path: str = ""
    keywords: list[str] = []
    train_override: dict[str, Any] = {}
    # 과거 이벤트 (WS 연결 안 된 상태에서 진행 로그 확인용)
    events: list[str] = []
    # 마지막 WS 이벤트 timestamp — 클라이언트가 "멈춤" 감지에 사용 (ISO8601)
    last_event_at: str | None = None


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    req: JobCreateRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """잡 생성 후 백그라운드에서 9-stage 파이프라인 실행."""
    from wfp_worker.stage1_init import GoalBuilder

    try:
        spec = GoalBuilder().build(
            ksic_section=req.ksic_section,
            task_type=req.task_type,
            description=req.description,
            ksic_division=req.ksic_division,
            language=req.language,
            user_id=user.id,
        )
    except WFPError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e

    # 사용자 입력 원본 보존 — 재실행 prefill 용
    if req.model_hf_repo:
        spec.model_hf_repo = req.model_hf_repo
    if req.data_source_mode:
        from wfp_worker.common.types import DataSourceMode
        spec.data_source_mode = DataSourceMode(req.data_source_mode)
    spec.data_source_path = req.data_source_path or ""
    spec.keywords = req.keywords or []
    spec.train_override = req.train_override or {}

    job_id = str(spec.job_id)
    _jobs[job_id] = spec
    _ws_clients[job_id] = []

    # req 보존 — advance 시 stage 재실행에 필요
    req_dict = req.model_dump()
    _job_reqs[job_id] = req_dict

    # async 라우터 안 → 현재 이벤트 루프에 태스크 등록 + 추적
    task = asyncio.create_task(_run_pipeline(spec, req_dict))
    _job_tasks[job_id] = task
    log.info("job_created", extra={"user_id": user.id, "job_id": job_id})

    return {"job_id": job_id, "state": spec.state.value}


# ─────────────────────────────────────────
# 사용자 → 관리자 알림 (PRD §10.5)
# 동적 라우트 /{job_id} 보다 위에 등록 — 정적 경로가 우선 매칭되도록.
# ─────────────────────────────────────────


class NotifyAdminRequest(BaseModel):
    """잡 실패 등 환경 문제 발생 시 사용자가 관리자에게 보내는 알림."""
    job_id: str | None = None
    kind: str = "other"            # env_missing / hw_mismatch / quota / other
    error_code: str | None = None
    message: str
    detail: dict[str, Any] = {}


@router.post("/notify-admin", status_code=status.HTTP_201_CREATED)
def notify_admin(
    body: NotifyAdminRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """현재 로그인한 사용자(USER/ADMIN 무관)가 관리자에게 알림 송신."""
    import json as _json
    n = AdminNotification(
        user_id=user.id,
        user_email=user.email,
        job_id=body.job_id,
        kind=body.kind,
        error_code=body.error_code,
        message=body.message,
        detail_json=_json.dumps(body.detail),
        status="new",
    )
    db.add(n)
    db.commit()
    log.info(
        "admin_notification_created",
        extra={"user_id": user.id, "extra": {"kind": body.kind, "code": body.error_code}},
    )
    return {"id": n.id, "status": "new"}


# ─────────────────────────────────────────
# 일괄 삭제 — /{job_id} 정적 경로보다 먼저 등록
# ─────────────────────────────────────────


class BulkDeleteRequest(BaseModel):
    """종료된 잡 일괄 삭제. states 미지정 시 FAILED 만."""
    states: list[str] = ["FAILED"]


@router.post("/bulk-delete")
def bulk_delete_jobs(
    body: BulkDeleteRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """본인 잡 중 지정 상태 일괄 삭제 (ADMIN은 전체)."""
    from wfp_worker.runtime import _job_events, save_state
    target_states: set[str] = set()
    for s in body.states:
        try:
            target_states.add(JobState(s).value)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid state: {s}") from None
    # 종료 상태만 허용
    allowed_terminal = {JobState.DONE.value, JobState.FAILED.value, JobState.CANCELLED.value}
    if not target_states.issubset(allowed_terminal):
        raise HTTPException(
            status_code=400,
            detail=f"bulk-delete는 종료 상태만 허용: {sorted(allowed_terminal)}",
        )
    is_admin = user.role.value == "ADMIN"
    to_delete = [
        jid for jid, spec in _jobs.items()
        if spec.state.value in target_states and (is_admin or spec.user_id == user.id)
    ]
    for jid in to_delete:
        _jobs.pop(jid, None)
        _ws_clients.pop(jid, None)
        _job_events.pop(jid, None)
    if to_delete:
        save_state()
    log.info("bulk_delete_jobs",
             extra={"user_id": user.id, "extra": {"count": len(to_delete), "states": list(target_states)}})
    return {"deleted": len(to_delete), "job_ids": to_delete}


@router.get("", response_model=list[JobListItem])
def list_jobs(
    user: User = Depends(get_current_user),
    state: str = "",
    limit: int = 50,
) -> list[JobListItem]:
    """현재 사용자의 잡 목록 (인메모리)."""
    items = []
    for spec in _jobs.values():
        if spec.user_id != user.id:
            continue
        if state and spec.state.value != state:
            continue
        items.append(JobListItem(
            job_id=str(spec.job_id),
            state=spec.state.value,
            current_stage=spec.current_stage,
            ksic_section=spec.ksic_section,
            task_type=spec.task_type.value,
            model_hf_repo=spec.model_hf_repo,
            created_at=spec.created_at.isoformat() if spec.created_at else None,
            error_code=spec.error_code,
        ))
    items.sort(key=lambda x: x.created_at or "", reverse=True)
    return items[:limit]


@router.get("/{job_id}", response_model=JobDetail)
def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> JobDetail:
    """잡 상태 조회 (소유자 또는 ADMIN만)."""
    spec = _get_owned_or_admin(job_id, user)
    return _to_detail(spec)


@router.delete("/{job_id}")
def delete_or_cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """진행 중 잡 → CANCELLED, 종료된 잡(FAILED/DONE/CANCELLED) → 영구 삭제.

    소유자 또는 ADMIN.
    """
    from wfp_worker.runtime import _job_events, save_state
    spec = _get_owned_or_admin(job_id, user)
    if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
        # 영구 삭제 — in-memory + persistence + WS clients + events 버퍼
        _jobs.pop(job_id, None)
        _ws_clients.pop(job_id, None)
        _job_events.pop(job_id, None)
        save_state()
        log.info("job_deleted", extra={"user_id": user.id, "job_id": job_id})
        return {"job_id": job_id, "state": "DELETED"}
    # 진행 중 → 취소
    spec.state = JobState.CANCELLED
    return {"job_id": job_id, "state": "CANCELLED"}


@router.post("/{job_id}/advance")
async def advance_stuck_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """
    멈춘 stage 강제 스킵 + 다음 stage 부터 재실행.

    소유자 또는 ADMIN 호출 가능. 종료된 잡(DONE/FAILED/CANCELLED) 은 거부.
    Stage 9 가 마지막 — 그 이상 advance 는 자동 DONE 마킹.

    주의: 스킵된 stage 의 산출물은 누락되어 후속 stage 에서 에러 가능. 사용 전 confirm 권장.
    """
    spec = _get_owned_or_admin(job_id, user)
    try:
        result = await advance_job(job_id, reason=f"user:{user.id}")
    except WFPError as e:
        # JOB_NOT_FOUND / INVALID_STATE
        raise HTTPException(status_code=409, detail=e.to_dict()) from e
    log.info("job_advance_invoked",
             extra={"user_id": user.id, "job_id": job_id, "extra": result})
    _ = spec  # silence unused after _get_owned_or_admin auth gate
    return result


@router.post("/{job_id}/finalize")
async def finalize_stuck_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """
    잡 조기 종료 — 현재 stage 까지 산출물로 DONE 마킹. 평가/패키징 스킵.

    체크포인트가 있으면 학습 결과만으로 사용/다운로드 가능.
    종료된 잡(DONE/FAILED/CANCELLED) 은 거부.
    """
    spec = _get_owned_or_admin(job_id, user)
    try:
        result = await finalize_job(job_id, reason=f"user:{user.id}")
    except WFPError as e:
        raise HTTPException(status_code=409, detail=e.to_dict()) from e
    log.info("job_finalize_invoked",
             extra={"user_id": user.id, "job_id": job_id, "extra": result})
    _ = spec
    return result


@router.get("/{job_id}/artifacts/{kind}")
def download_artifact(
    job_id: str,
    kind: str,
    user: User = Depends(get_current_user),
) -> FileResponse:
    """아티팩트 파일 다운로드. kind: adapter / merged / gguf / zip / model_card."""
    spec = _get_owned_or_admin(job_id, user)
    try:
        target_kind = ArtifactKind(kind)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid kind: {kind}") from None

    artifact = next((a for a in spec.artifacts if a.kind == target_kind), None)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact '{kind}' not found")

    path = Path(artifact.uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")

    # 디렉토리 (adapter/merged 등) — 즉석에서 zip 생성 후 전송
    if path.is_dir():
        import shutil
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        # shutil.make_archive는 .zip 확장자 자동 추가 → base_name에서 제거
        base = tmp.name[:-4] if tmp.name.endswith(".zip") else tmp.name
        shutil.make_archive(base, "zip", root_dir=str(path))
        zip_path = f"{base}.zip"
        return FileResponse(
            path=zip_path,
            filename=f"{path.name}.zip",
            media_type="application/zip",
            background=None,  # 응답 후 tmp 파일 정리는 OS에 위임 (작은 수)
        )

    # 단일 파일 — Content-Type 확장자 기반
    media_map = {
        ".md": "text/markdown",
        ".json": "application/json",
        ".gguf": "application/octet-stream",
        ".safetensors": "application/octet-stream",
        ".bin": "application/octet-stream",
        ".zip": "application/zip",
    }
    media = media_map.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(path), filename=path.name, media_type=media)


@router.get("/{job_id}/data/raw")
def download_raw_data(
    job_id: str,
    user: User = Depends(get_current_user),
) -> FileResponse:
    """
    Stage 5 수집 직후 원시 데이터(JSONL) 다운로드.

    경로: ~/.wfp/jobs/<job_id>/raw/collected.jsonl
    모든 data_source_mode (UPLOAD/DIRECTORY/WEB) 동일 위치.
    SYNTHETIC 은 수집 단계가 없어 404.
    소유자 또는 ADMIN.
    """
    spec = _get_owned_or_admin(job_id, user)

    raw_path = Path.home() / ".wfp" / "jobs" / str(spec.job_id) / "raw" / "collected.jsonl"
    if not raw_path.exists():
        raise HTTPException(
            status_code=404,
            detail="수집된 원시 데이터 없음. 잡이 Stage 5(데이터 수집)에 진입했는지 확인하세요.",
        )

    return FileResponse(
        path=str(raw_path),
        filename=f"{job_id[:8]}-collected.jsonl",
        media_type="application/x-ndjson",
    )


class HFUploadRequest(BaseModel):
    """파인튜닝 모델을 HuggingFace Hub에 푸시."""
    repo_id: str        # "username/repo-name"
    artifact_kind: str = "adapter"   # adapter / merged
    private: bool = True
    hf_token: str = ""  # 비어있으면 user.hf_token_encrypted 사용


@router.post("/{job_id}/upload-hf")
def upload_to_hf(
    job_id: str,
    body: HFUploadRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """잡 완료된 모델을 HF Hub로 업로드.

    토큰 우선순위: body.hf_token > user.hf_token_encrypted (settings에서 저장).
    """
    spec = _get_owned_or_admin(job_id, user)
    if spec.state != JobState.DONE:
        raise HTTPException(status_code=409, detail=f"Job not DONE (current: {spec.state.value})")

    try:
        target_kind = ArtifactKind(body.artifact_kind)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid artifact_kind: {body.artifact_kind}") from None

    artifact = next((a for a in spec.artifacts if a.kind == target_kind), None)
    if not artifact:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{body.artifact_kind}' 없음. adapter/merged 등 디렉토리 형식만 업로드 가능.",
        )

    artifact_path = Path(artifact.uri)
    if not artifact_path.exists() or not artifact_path.is_dir():
        raise HTTPException(status_code=400, detail="업로드는 디렉토리 형식만 지원 (adapter/merged).")

    # 토큰 결정 — hf_token_encrypted는 현재 평문 저장 (users.py TODO: encrypt)
    token = body.hf_token.strip()
    if not token and user.hf_token_encrypted:
        token = user.hf_token_encrypted
    if not token:
        raise HTTPException(
            status_code=400,
            detail="HF 토큰 없음. 설정 페이지에서 토큰 저장 또는 body.hf_token 명시.",
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="huggingface_hub 미설치. 워커 venv: uv pip install -U huggingface_hub",
        ) from None

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=body.repo_id, repo_type="model", private=body.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(artifact_path),
            repo_id=body.repo_id,
            repo_type="model",
            commit_message=f"WFP fine-tune job {job_id[:8]} ({body.artifact_kind})",
        )
    except Exception as e:
        log.exception("HF upload failed", extra={"user_id": user.id, "job_id": job_id})
        raise HTTPException(status_code=502, detail=f"HF 업로드 실패: {e}") from e

    log.info("hf_upload_success",
             extra={"user_id": user.id, "job_id": job_id,
                    "extra": {"repo": body.repo_id, "kind": body.artifact_kind}})
    return {
        "status": "ok",
        "repo_id": body.repo_id,
        "repo_url": f"https://huggingface.co/{body.repo_id}",
        "private": body.private,
    }


# ─────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────


def _get_owned_or_admin(job_id: str, user: User) -> JobSpec:
    spec = _jobs.get(job_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Job not found")
    if spec.user_id != user.id and user.role.value != "ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return spec


def _to_detail(spec: JobSpec) -> JobDetail:
    from wfp_worker.runtime import get_job_events
    return JobDetail(
        job_id=str(spec.job_id),
        state=spec.state.value,
        current_stage=spec.current_stage,
        ksic_section=spec.ksic_section,
        task_type=spec.task_type.value,
        model_hf_repo=spec.model_hf_repo,
        created_at=spec.created_at.isoformat() if spec.created_at else None,
        error_code=spec.error_code,
        error_message=spec.error_message,
        num_train=spec.num_train,
        num_val=spec.num_val,
        num_test=spec.num_test,
        artifacts=[a.model_dump(mode="json") for a in spec.artifacts],
        # 재실행 prefill용 — 사용자 입력 원본 그대로 반환
        ksic_division=spec.ksic_division,
        description=spec.description,
        language=spec.language,
        data_source_mode=spec.data_source_mode.value,
        data_source_path=spec.data_source_path,  # 원본 사용자 입력 (dataset_path는 정제 결과)
        keywords=spec.keywords,
        train_override=spec.train_override,
        events=get_job_events(str(spec.job_id)),
        last_event_at=get_last_event_at(str(spec.job_id)),
    )


# ─────────────────────────────────────────
# 사용자 → 관리자 알림 (PRD §10.5)
# ─────────────────────────────────────────


