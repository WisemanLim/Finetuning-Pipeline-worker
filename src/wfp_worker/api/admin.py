"""
관리자 라우터 (ADMIN 권한 필요).

엔드포인트 (PRD §10):
  10.1 대시보드
    GET   /admin/stats                통계 카드 (사용자/잡/리소스/타이밍)
    GET   /admin/resources            GPU/디스크/큐 현황

  10.2 작업 모니터링
    GET   /admin/jobs                 전체 잡 목록 (필터 지원)
    GET   /admin/jobs/{id}            잡 상세 + 이벤트 로그
    POST  /admin/jobs/{id}/cancel     강제 취소
    PATCH /admin/jobs/{id}/priority   우선순위 변경

  10.3 자원 회수
    POST  /admin/reaper/run           Reaper 수동 트리거

  10.4 카탈로그 관리
    GET/POST/PATCH/DELETE /admin/catalog/models      모델 카탈로그 CRUD
    GET/POST/PATCH/DELETE /admin/data-sources        데이터 소스 화이트리스트 CRUD
    GET/POST/PATCH/DELETE /admin/presets             카테고리/프리셋 CRUD

  10.5 감사 로그
    GET   /admin/audit                감사 로그 (필터 지원)

  부가
    GET   /admin/users                사용자 목록
    PATCH /admin/users/{id}           권한/활성화 토글
    GET   /admin/env/status           환경 진단
    POST  /admin/env/install          학습 패키지 자동 설치
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from wfp_worker.auth import get_current_admin
from wfp_worker.db import (
    AuditLog,
    CategoryPreset,
    DataSourceWhitelist,
    ModelCatalogEntry,
    User,
    UserRole,
    get_db,
)
from wfp_worker.runtime import _jobs

router = APIRouter(prefix="/admin", tags=["admin"])


# ─────────────────────────────────────────
# Audit 헬퍼
# ─────────────────────────────────────────


def _audit(db: Session, actor: User, action: str, target: str = "", detail: str = "") -> None:
    db.add(AuditLog(
        actor_id=actor.id, actor_email=actor.email,
        action=action, target=target, detail=detail,
    ))
    db.commit()


# ─────────────────────────────────────────
# Users
# ─────────────────────────────────────────


class AdminUserItem(BaseModel):
    id: str
    email: str
    name: str | None
    role: str
    is_active: bool
    last_login: str | None
    created_at: str


class UserPatch(BaseModel):
    role: str | None = None
    is_active: bool | None = None


@router.get("/users", response_model=list[AdminUserItem])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[AdminUserItem]:
    rows = db.query(User).order_by(User.created_at.desc()).all()
    return [
        AdminUserItem(
            id=u.id, email=u.email, name=u.name, role=u.role.value, is_active=u.is_active,
            last_login=u.last_login.isoformat() if u.last_login else None,
            created_at=u.created_at.isoformat() if u.created_at else "",
        )
        for u in rows
    ]


@router.patch("/users/{user_id}", response_model=AdminUserItem)
def patch_user(
    user_id: str,
    req: UserPatch,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> AdminUserItem:
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    changes = {}
    if req.role:
        try:
            u.role = UserRole(req.role.upper())
            changes["role"] = req.role
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid role: {req.role}") from None
    if req.is_active is not None:
        u.is_active = req.is_active
        changes["is_active"] = req.is_active

    db.commit()
    _audit(db, actor, "user_patch", target=user_id, detail=str(changes))
    return AdminUserItem(
        id=u.id, email=u.email, name=u.name, role=u.role.value, is_active=u.is_active,
        last_login=u.last_login.isoformat() if u.last_login else None,
        created_at=u.created_at.isoformat() if u.created_at else "",
    )


# ─────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────


class AdminJobItem(BaseModel):
    job_id: str
    user_id: str
    state: str
    current_stage: int
    ksic_section: str
    model_hf_repo: str
    created_at: str | None


@router.get("/jobs", response_model=list[AdminJobItem])
def list_all_jobs(
    state: str = "",
    user_id: str = "",
    ksic_section: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 200,
    _: User = Depends(get_current_admin),
) -> list[AdminJobItem]:
    dt_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc) if date_from else None
    dt_to = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) if date_to else None

    items = []
    for spec in _jobs.values():
        if state and spec.state.value != state:
            continue
        if user_id and spec.user_id != user_id:
            continue
        if ksic_section and spec.ksic_section != ksic_section:
            continue
        if dt_from and spec.created_at and spec.created_at < dt_from:
            continue
        if dt_to and spec.created_at and spec.created_at > dt_to:
            continue
        items.append(AdminJobItem(
            job_id=str(spec.job_id), user_id=spec.user_id,
            state=spec.state.value, current_stage=spec.current_stage,
            ksic_section=spec.ksic_section, model_hf_repo=spec.model_hf_repo,
            created_at=spec.created_at.isoformat() if spec.created_at else None,
        ))
    items.sort(key=lambda x: x.created_at or "", reverse=True)
    return items[:limit]


class AdminJobDetail(AdminJobItem):
    error_message: str = ""
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    artifacts: list[dict] = []
    events: list[str] = []


@router.get("/jobs/{job_id}", response_model=AdminJobDetail)
def get_admin_job(
    job_id: str,
    _: User = Depends(get_current_admin),
) -> AdminJobDetail:
    from wfp_worker.runtime import get_job_events

    spec = _jobs.get(job_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Job not found")
    return AdminJobDetail(
        job_id=str(spec.job_id), user_id=spec.user_id,
        state=spec.state.value, current_stage=spec.current_stage,
        ksic_section=spec.ksic_section, model_hf_repo=spec.model_hf_repo,
        created_at=spec.created_at.isoformat() if spec.created_at else None,
        error_message=spec.error_message,
        num_train=spec.num_train, num_val=spec.num_val, num_test=spec.num_test,
        artifacts=[a.model_dump(mode="json") for a in spec.artifacts],
        events=get_job_events(job_id),
    )


@router.post("/jobs/{job_id}/cancel")
def force_cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> dict:
    from wfp_worker.common.types import JobState

    spec = _jobs.get(job_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Job not found")
    if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
        raise HTTPException(status_code=409, detail=f"Job already {spec.state.value}")
    spec.state = JobState.CANCELLED
    spec.finished_at = datetime.now(timezone.utc)
    _audit(db, actor, "job_cancel", target=job_id, detail=f"user_id={spec.user_id}")
    return {"job_id": job_id, "state": "CANCELLED"}


class PriorityUpdate(BaseModel):
    priority: int = Field(ge=1, le=9, description="1=가장 높음, 9=가장 낮음. 기본 5")


@router.patch("/jobs/{job_id}/priority", response_model=AdminJobItem)
def update_job_priority(
    job_id: str,
    body: PriorityUpdate,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> AdminJobItem:
    """잡 우선순위 변경 (PRD §10.2). PENDING 상태에서만 의미가 있음."""
    spec = _jobs.get(job_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Job not found")
    old = spec.priority
    spec.priority = body.priority
    _audit(
        db, actor, "job_priority",
        target=job_id, detail=f"{old} -> {body.priority}",
    )
    return AdminJobItem(
        job_id=str(spec.job_id), user_id=spec.user_id,
        state=spec.state.value, current_stage=spec.current_stage,
        ksic_section=spec.ksic_section, model_hf_repo=spec.model_hf_repo,
        created_at=spec.created_at.isoformat() if spec.created_at else None,
    )


# ─────────────────────────────────────────
# Resources
# ─────────────────────────────────────────


class GpuStatus(BaseModel):
    index: int
    name: str
    vendor: str
    vram_gb: float
    utilization_pct: float
    temperature_c: float
    job_id: str | None = None


class ResourcesResponse(BaseModel):
    gpus: list[GpuStatus]
    queue_pending: int
    queue_running: int
    disk_free_gb: float


@router.get("/resources", response_model=ResourcesResponse)
def get_resources(_: User = Depends(get_current_admin)) -> ResourcesResponse:
    from wfp_worker.common.types import JobState
    from wfp_worker.stage2_hw_probe import HWProber

    caps = HWProber().probe()
    pending = sum(1 for s in _jobs.values() if s.state == JobState.PENDING)
    running = sum(1 for s in _jobs.values() if s.state in (JobState.RUNNING, JobState.EVAL))

    return ResourcesResponse(
        gpus=[GpuStatus(
            index=g.index, name=g.name, vendor=g.vendor, vram_gb=g.vram_gb,
            utilization_pct=g.utilization_pct, temperature_c=g.temperature_c,
        ) for g in caps.gpus],
        queue_pending=pending,
        queue_running=running,
        disk_free_gb=caps.disk_free_gb,
    )


# ─────────────────────────────────────────
# Catalog CRUD
# ─────────────────────────────────────────


class CatalogItem(BaseModel):
    id: str
    hf_repo: str
    params_b: float
    license: str
    hw_min_vram_gb: float
    language: list[str]
    tags: list[str]
    priority: int
    is_active: bool


class CatalogCreate(BaseModel):
    hf_repo: str
    params_b: float = Field(gt=0)
    license: str = "apache-2.0"
    hw_min_vram_gb: float = Field(ge=0)
    language: list[str] = ["ko", "en"]
    tags: list[str] = ["general"]
    priority: int = 5


class CatalogPatch(BaseModel):
    params_b: float | None = None
    license: str | None = None
    hw_min_vram_gb: float | None = None
    language: list[str] | None = None
    tags: list[str] | None = None
    priority: int | None = None
    is_active: bool | None = None


def _to_catalog_item(e: ModelCatalogEntry) -> CatalogItem:
    return CatalogItem(
        id=e.id, hf_repo=e.hf_repo, params_b=e.params_b, license=e.license,
        hw_min_vram_gb=e.hw_min_vram_gb,
        language=[s.strip() for s in e.language.split(",") if s.strip()],
        tags=[s.strip() for s in e.tags.split(",") if s.strip()],
        priority=e.priority, is_active=e.is_active,
    )


@router.get("/catalog/models", response_model=list[CatalogItem])
def list_catalog(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[CatalogItem]:
    rows = db.query(ModelCatalogEntry).order_by(ModelCatalogEntry.priority, ModelCatalogEntry.hf_repo).all()
    return [_to_catalog_item(e) for e in rows]


@router.post("/catalog/models", response_model=CatalogItem, status_code=status.HTTP_201_CREATED)
def create_catalog(
    req: CatalogCreate,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> CatalogItem:
    if db.query(ModelCatalogEntry).filter(ModelCatalogEntry.hf_repo == req.hf_repo).first():
        raise HTTPException(status_code=409, detail="Already exists")
    entry = ModelCatalogEntry(
        hf_repo=req.hf_repo, params_b=req.params_b, license=req.license,
        hw_min_vram_gb=req.hw_min_vram_gb,
        language=",".join(req.language), tags=",".join(req.tags), priority=req.priority,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "catalog_create", target=entry.id, detail=req.hf_repo)
    return _to_catalog_item(entry)


@router.patch("/catalog/models/{entry_id}", response_model=CatalogItem)
def patch_catalog(
    entry_id: str,
    req: CatalogPatch,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> CatalogItem:
    entry = db.get(ModelCatalogEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    for field, value in req.model_dump(exclude_none=True).items():
        if field in ("language", "tags"):
            value = ",".join(value)
        setattr(entry, field, value)
    db.commit()
    _audit(db, actor, "catalog_patch", target=entry_id, detail=str(req.model_dump(exclude_none=True)))
    db.refresh(entry)
    return _to_catalog_item(entry)


@router.delete("/catalog/models/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_catalog(
    entry_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> None:
    """soft delete — is_active=False."""
    entry = db.get(ModelCatalogEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    entry.is_active = False
    db.commit()
    _audit(db, actor, "catalog_delete", target=entry_id, detail=entry.hf_repo)


# ─────────────────────────────────────────
# Stats (대시보드)
# ─────────────────────────────────────────


class GpuBreakdown(BaseModel):
    index: int
    name: str
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    util_pct: float = 0.0


class StatsResponse(BaseModel):
    """PRD §10.1 대시보드 카드."""

    users_total: int
    users_active: int
    jobs_pending: int
    jobs_running: int
    jobs_done: int
    jobs_failed: int
    jobs_cancelled: int
    queue_depth: int                   # PENDING 잡 개수 = 큐 길이
    avg_wait_seconds: float | None = None    # PENDING → RUNNING 평균
    avg_train_seconds: float | None = None   # RUNNING → DONE 평균
    disk_free_gb: float
    gpu_count: int
    gpu_breakdown: list[GpuBreakdown] = []


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> StatsResponse:
    from wfp_worker.common.types import JobState
    from wfp_worker.stage2_hw_probe import HWProber

    users_total = db.query(User).count()
    users_active = db.query(User).filter(User.is_active.is_(True)).count()

    pending = running = done = failed = cancelled = 0
    wait_durations: list[float] = []
    train_durations: list[float] = []

    for spec in _jobs.values():
        s = spec.state
        if s == JobState.PENDING:
            pending += 1
        elif s in (JobState.RUNNING, JobState.EVAL):
            running += 1
        elif s == JobState.DONE:
            done += 1
        elif s == JobState.FAILED:
            failed += 1
        elif s == JobState.CANCELLED:
            cancelled += 1

        # 타이밍 누적 (started_at/finished_at 모두 존재하는 잡만)
        if spec.started_at and spec.created_at:
            wait_durations.append((spec.started_at - spec.created_at).total_seconds())
        if spec.finished_at and spec.started_at and s == JobState.DONE:
            train_durations.append((spec.finished_at - spec.started_at).total_seconds())

    avg_wait = sum(wait_durations) / len(wait_durations) if wait_durations else None
    avg_train = sum(train_durations) / len(train_durations) if train_durations else None

    caps = HWProber().probe()
    gpu_breakdown = [
        GpuBreakdown(
            index=i, name=getattr(g, "name", f"GPU-{i}"),
            vram_used_mb=getattr(g, "vram_used_mb", 0.0),
            vram_total_mb=getattr(g, "vram_total_mb", 0.0),
            util_pct=getattr(g, "util_pct", 0.0),
        )
        for i, g in enumerate(caps.gpus)
    ]

    return StatsResponse(
        users_total=users_total, users_active=users_active,
        jobs_pending=pending, jobs_running=running,
        jobs_done=done, jobs_failed=failed, jobs_cancelled=cancelled,
        queue_depth=pending,
        avg_wait_seconds=avg_wait, avg_train_seconds=avg_train,
        disk_free_gb=caps.disk_free_gb, gpu_count=len(caps.gpus),
        gpu_breakdown=gpu_breakdown,
    )


# ─────────────────────────────────────────
# 환경 진단 + 자동 설치
# ─────────────────────────────────────────


@router.get("/env/status")
def env_status(_: User = Depends(get_current_admin)) -> dict:
    """현재 환경 진단 — 프로파일, 누락 패키지, 설치 명령."""
    from wfp_worker.env_check import diagnose
    return diagnose().to_dict()


@router.post("/env/install")
def env_install(
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> dict:
    """누락 패키지 자동 설치 (subprocess uv pip install)."""
    from wfp_worker.env_check import install_missing
    result = install_missing()
    _audit(db, actor, "env_install", detail=f"status={result.get('status')}")
    return result


# ─────────────────────────────────────────
# Audit
# ─────────────────────────────────────────


class AuditItem(BaseModel):
    id: str
    ts: str
    actor_email: str
    action: str
    target: str | None
    detail: str | None


@router.get("/audit", response_model=list[AuditItem])
def list_audit(
    limit: int = 100,
    actor_email: str = "",
    action: str = "",
    date_from: str = "",
    date_to: str = "",
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[AuditItem]:
    q = db.query(AuditLog)
    if actor_email:
        q = q.filter(AuditLog.actor_email.ilike(f"%{actor_email}%"))
    if action:
        q = q.filter(AuditLog.action == action)
    if date_from:
        q = q.filter(AuditLog.ts >= datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc))
    if date_to:
        q = q.filter(AuditLog.ts <= datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc))
    rows = q.order_by(AuditLog.ts.desc()).limit(limit).all()
    return [
        AuditItem(
            id=r.id, ts=r.ts.isoformat(), actor_email=r.actor_email,
            action=r.action, target=r.target, detail=r.detail,
        )
        for r in rows
    ]


# ─────────────────────────────────────────
# Reaper — 자원 회수 수동 트리거 (PRD §10.3)
# ─────────────────────────────────────────


class ReaperResult(BaseModel):
    jobs_failed: int
    artifacts_deleted: int
    checkpoints_purged: int
    disk_freed_mb: float
    detail: list[str]


@router.post("/reaper/run", response_model=ReaperResult)
def run_reaper(
    heartbeat_timeout_min: int = 30,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> ReaperResult:
    """
    Reaper 수동 실행.
    - heartbeat_timeout_min 이상 진행 없는 RUNNING 잡 → FAILED
    - 디스크 임계치 초과 시 오래된 체크포인트 회수
    """
    import shutil
    from pathlib import Path

    from wfp_worker.common.types import JobState  # noqa: PLC0415
    from wfp_worker.runtime import _job_events, save_state  # noqa: PLC0415

    now = datetime.now(timezone.utc)
    jobs_failed = 0
    artifacts_deleted = 0
    checkpoints_purged = 0
    disk_freed = 0.0
    detail: list[str] = []

    # 1. heartbeat 타임아웃 — created_at 기준 N분 초과 RUNNING 잡 → FAILED
    for jid, spec in list(_jobs.items()):
        if spec.state not in (JobState.RUNNING, JobState.EVAL):
            continue
        if not _job_events.get(jid):
            continue
        elapsed = (now - spec.created_at).total_seconds() if spec.created_at else 0
        if elapsed > heartbeat_timeout_min * 60:
            spec.state = JobState.FAILED
            spec.error_message = f"Reaper: heartbeat timeout ({heartbeat_timeout_min}min)"
            jobs_failed += 1
            detail.append(f"job {jid} → FAILED (timeout)")

    # 2. 오래된 체크포인트 회수 (디스크 임계치 초과 시)
    wfp_dir = Path("/data") if Path("/data").exists() else Path.home() / ".wfp"
    ckpt_dir = wfp_dir / "checkpoints"
    if ckpt_dir.exists():
        total, used, _ = shutil.disk_usage(wfp_dir)
        used_pct = used / total * 100 if total else 0
        if used_pct > 80:
            job_ckpts = sorted(ckpt_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            for ckpt in job_ckpts[:max(1, len(job_ckpts) // 2)]:
                jid = ckpt.name
                if jid in _jobs and _jobs[jid].state in (JobState.RUNNING, JobState.EVAL):
                    continue
                size_bytes = sum(
                    f.stat().st_size for f in ckpt.rglob("*") if f.is_file()
                )
                size_mb = size_bytes / 1024 / 1024
                shutil.rmtree(ckpt, ignore_errors=True)
                checkpoints_purged += 1
                disk_freed += size_mb
                detail.append(f"checkpoint {jid} purged ({size_mb:.1f}MB)")

    save_state()
    _audit(db, actor, "reaper_run", detail=f"failed={jobs_failed} purged={checkpoints_purged}")
    return ReaperResult(
        jobs_failed=jobs_failed,
        artifacts_deleted=artifacts_deleted,
        checkpoints_purged=checkpoints_purged,
        disk_freed_mb=round(disk_freed, 2),
        detail=detail,
    )


# ─────────────────────────────────────────
# 데이터 소스 화이트리스트 (PRD §10.4)
# ─────────────────────────────────────────


class DataSourceItem(BaseModel):
    id: str
    name: str
    source_type: str
    url_pattern: str
    description: str | None = None
    is_active: bool


class DataSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source_type: str = Field(pattern="^(hf_datasets|url|api|upload)$")
    url_pattern: str = Field(min_length=1, max_length=512)
    description: str | None = None


class DataSourcePatch(BaseModel):
    name: str | None = None
    source_type: str | None = Field(default=None, pattern="^(hf_datasets|url|api|upload)$")
    url_pattern: str | None = None
    description: str | None = None
    is_active: bool | None = None


def _to_ds(d: DataSourceWhitelist) -> DataSourceItem:
    return DataSourceItem(
        id=d.id, name=d.name, source_type=d.source_type,
        url_pattern=d.url_pattern, description=d.description,
        is_active=d.is_active,
    )


@router.get("/data-sources", response_model=list[DataSourceItem])
def list_data_sources(
    active_only: bool = False,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[DataSourceItem]:
    q = db.query(DataSourceWhitelist)
    if active_only:
        q = q.filter(DataSourceWhitelist.is_active.is_(True))
    return [_to_ds(d) for d in q.order_by(DataSourceWhitelist.created_at.desc()).all()]


@router.post("/data-sources", response_model=DataSourceItem, status_code=status.HTTP_201_CREATED)
def create_data_source(
    body: DataSourceCreate,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> DataSourceItem:
    if db.query(DataSourceWhitelist).filter_by(name=body.name).first():
        raise HTTPException(status_code=409, detail="Name already exists")
    entry = DataSourceWhitelist(**body.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "datasource_create", target=entry.id, detail=entry.name)
    return _to_ds(entry)


@router.patch("/data-sources/{entry_id}", response_model=DataSourceItem)
def patch_data_source(
    entry_id: str,
    body: DataSourcePatch,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> DataSourceItem:
    entry = db.get(DataSourceWhitelist, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(entry, k, v)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "datasource_patch", target=entry_id, detail=entry.name)
    return _to_ds(entry)


@router.delete("/data-sources/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_data_source(
    entry_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> None:
    entry = db.get(DataSourceWhitelist, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    entry.is_active = False
    db.commit()
    _audit(db, actor, "datasource_delete", target=entry_id, detail=entry.name)


# ─────────────────────────────────────────
# 카테고리/프리셋 (PRD §10.4)
# ─────────────────────────────────────────


class PresetItem(BaseModel):
    id: str
    name: str
    ksic_section: str
    task_type: str
    recommended_models: list[str]
    default_config: dict
    description: str | None = None
    is_active: bool


class PresetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    ksic_section: str = Field(min_length=1, max_length=2)
    task_type: str = Field(min_length=1, max_length=32)
    recommended_models: list[str] = []
    default_config: dict = {}
    description: str | None = None


class PresetPatch(BaseModel):
    name: str | None = None
    ksic_section: str | None = None
    task_type: str | None = None
    recommended_models: list[str] | None = None
    default_config: dict | None = None
    description: str | None = None
    is_active: bool | None = None


def _to_preset(p: CategoryPreset) -> PresetItem:
    return PresetItem(
        id=p.id, name=p.name, ksic_section=p.ksic_section, task_type=p.task_type,
        recommended_models=[m for m in p.recommended_models.split(",") if m],
        default_config=json.loads(p.default_config_json or "{}"),
        description=p.description, is_active=p.is_active,
    )


@router.get("/presets", response_model=list[PresetItem])
def list_presets(
    ksic_section: str = "",
    task_type: str = "",
    active_only: bool = False,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[PresetItem]:
    q = db.query(CategoryPreset)
    if ksic_section:
        q = q.filter(CategoryPreset.ksic_section == ksic_section)
    if task_type:
        q = q.filter(CategoryPreset.task_type == task_type)
    if active_only:
        q = q.filter(CategoryPreset.is_active.is_(True))
    return [_to_preset(p) for p in q.order_by(CategoryPreset.created_at.desc()).all()]


@router.post("/presets", response_model=PresetItem, status_code=status.HTTP_201_CREATED)
def create_preset(
    body: PresetCreate,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> PresetItem:
    if db.query(CategoryPreset).filter_by(name=body.name).first():
        raise HTTPException(status_code=409, detail="Name already exists")
    entry = CategoryPreset(
        name=body.name, ksic_section=body.ksic_section, task_type=body.task_type,
        recommended_models=",".join(body.recommended_models),
        default_config_json=json.dumps(body.default_config),
        description=body.description,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "preset_create", target=entry.id, detail=entry.name)
    return _to_preset(entry)


@router.patch("/presets/{entry_id}", response_model=PresetItem)
def patch_preset(
    entry_id: str,
    body: PresetPatch,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> PresetItem:
    entry = db.get(CategoryPreset, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    data = body.model_dump(exclude_unset=True)
    if "recommended_models" in data:
        entry.recommended_models = ",".join(data.pop("recommended_models"))
    if "default_config" in data:
        entry.default_config_json = json.dumps(data.pop("default_config"))
    for k, v in data.items():
        setattr(entry, k, v)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "preset_patch", target=entry_id, detail=entry.name)
    return _to_preset(entry)


@router.delete("/presets/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(
    entry_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> None:
    entry = db.get(CategoryPreset, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    entry.is_active = False
    db.commit()
    _audit(db, actor, "preset_delete", target=entry_id, detail=entry.name)
