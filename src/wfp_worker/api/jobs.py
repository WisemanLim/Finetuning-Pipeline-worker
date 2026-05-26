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
from wfp_worker.db import User, get_db

log = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


# 잡 인메모리 스토어 — server.py와 공유 (모듈 임포트 시 동일 인스턴스).
from wfp_worker.runtime import _jobs, _run_pipeline, _ws_clients  # noqa: E402


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

    # async 라우터 안 → 현재 이벤트 루프에 태스크 등록
    asyncio.create_task(_run_pipeline(spec, req.model_dump()))
    log.info("job_created", extra={"user_id": user.id, "job_id": job_id})

    return {"job_id": job_id, "state": spec.state.value}


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
def cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """잡 취소 (소유자 또는 ADMIN)."""
    spec = _get_owned_or_admin(job_id, user)
    if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
        raise HTTPException(status_code=409, detail=f"Job already {spec.state.value}")
    spec.state = JobState.CANCELLED
    return {"job_id": job_id, "state": "CANCELLED"}


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

    # 디렉토리(merged 등)는 클라이언트가 zip을 받도록 안내
    if path.is_dir():
        raise HTTPException(status_code=400, detail="Use kind=zip for directory artifacts")

    return FileResponse(path=str(path), filename=path.name, media_type="application/octet-stream")


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
    )
