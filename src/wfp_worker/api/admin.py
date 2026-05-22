"""
관리자 라우터 (ADMIN 권한 필요).

엔드포인트:
  GET   /admin/users                사용자 목록
  PATCH /admin/users/{id}           권한 변경 / 활성화 토글
  GET   /admin/jobs                 전체 잡 목록
  GET   /admin/resources            GPU/디스크/큐 현황
  GET   /admin/catalog/models       카탈로그 전체 (관리)
  POST  /admin/catalog/models       카탈로그 추가
  PATCH /admin/catalog/models/{id}  카탈로그 수정
  DELETE /admin/catalog/models/{id} 카탈로그 비활성
  GET   /admin/audit                감사 로그
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from wfp_worker.auth import get_current_admin
from wfp_worker.db import AuditLog, ModelCatalogEntry, User, UserRole, get_db
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
    _audit(db, actor, "job_cancel", target=job_id, detail=f"user_id={spec.user_id}")
    return {"job_id": job_id, "state": "CANCELLED"}


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


class StatsResponse(BaseModel):
    users_total: int
    users_active: int
    jobs_pending: int
    jobs_running: int
    jobs_done: int
    jobs_failed: int
    jobs_cancelled: int
    disk_free_gb: float
    gpu_count: int


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

    caps = HWProber().probe()
    return StatsResponse(
        users_total=users_total, users_active=users_active,
        jobs_pending=pending, jobs_running=running,
        jobs_done=done, jobs_failed=failed, jobs_cancelled=cancelled,
        disk_free_gb=caps.disk_free_gb, gpu_count=len(caps.gpus),
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
