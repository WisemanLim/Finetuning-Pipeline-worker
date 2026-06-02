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
    POST  /admin/catalog/seed-defaults               기본 시드 (1회용)
    POST  /admin/catalog/auto-refresh                HF Hub trending 기반 자동 갱신

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
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from wfp_worker.auth import get_current_admin
from wfp_worker.common.logger import get_logger
from wfp_worker.db import (
    AdminNotification,
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
log = get_logger(__name__)


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
    user_email: str = ""        # UUID 대신 표시용 이메일
    user_name: str = ""         # User.name (옵션 — 비어있을 수 있음)
    state: str
    current_stage: int
    ksic_section: str
    task_type: str = ""
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
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[AdminJobItem]:
    dt_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc) if date_from else None
    dt_to = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) if date_to else None

    # 한번에 user_id → email/name lookup (N+1 쿼리 회피)
    user_ids = {spec.user_id for spec in _jobs.values() if spec.user_id}
    user_map: dict[str, tuple[str, str]] = {}  # id → (email, name)
    if user_ids:
        for u in db.query(User).filter(User.id.in_(user_ids)).all():
            user_map[u.id] = (u.email, u.name or "")

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
        email, name = user_map.get(spec.user_id, ("", ""))
        items.append(AdminJobItem(
            job_id=str(spec.job_id), user_id=spec.user_id,
            user_email=email, user_name=name,
            state=spec.state.value, current_stage=spec.current_stage,
            ksic_section=spec.ksic_section,
            task_type=spec.task_type.value if hasattr(spec.task_type, "value") else str(spec.task_type),
            model_hf_repo=spec.model_hf_repo,
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
    is_active: bool  # False = HF 사용불가(미지원/삭제) 마킹
    downloaded: bool = False        # 로컬 HF cache 또는 local_path 존재
    downloading: bool = False       # 백그라운드 다운로드 진행 중
    source: str = "hf"              # "hf" / "finetuned" / "upload"
    is_public: bool = False         # source!="hf" 일 때 공개 여부
    local_path: str | None = None   # finetuned/upload 의 로컬 경로
    owner_user_id: str | None = None


class CatalogCreate(BaseModel):
    hf_repo: str
    params_b: float = Field(gt=0)
    license: str = "apache-2.0"
    hw_min_vram_gb: float = Field(ge=0)
    language: list[str] = ["ko", "en"]
    tags: list[str] = ["general"]
    priority: int = 5
    source: str = "hf"               # "hf" / "upload"
    local_path: str | None = None    # upload 시 서버 경로


class CatalogPatch(BaseModel):
    params_b: float | None = None
    license: str | None = None
    hw_min_vram_gb: float | None = None
    language: list[str] | None = None
    tags: list[str] | None = None
    priority: int | None = None
    is_active: bool | None = None
    is_public: bool | None = None


# 백그라운드 다운로드 추적 — repo_id → {started_at, error}
_catalog_downloading: dict[str, dict] = {}


def _is_model_downloaded(repo_id: str, local_path: str | None = None) -> bool:
    """다운로드 완료 여부.

    - source=hf: HF cache 에 config.json 존재
    - source=finetuned/upload: local_path 디렉토리 존재
    """
    if local_path:
        try:
            from pathlib import Path as _P
            return _P(local_path).exists()
        except Exception:
            return False
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(repo_id, "config.json")
        return bool(path)
    except Exception:
        return False


def _to_catalog_item(e: ModelCatalogEntry) -> CatalogItem:
    source = getattr(e, "source", "hf") or "hf"
    local_path = getattr(e, "local_path", None)
    return CatalogItem(
        id=e.id, hf_repo=e.hf_repo, params_b=e.params_b, license=e.license,
        hw_min_vram_gb=e.hw_min_vram_gb,
        language=[s.strip() for s in e.language.split(",") if s.strip()],
        tags=[s.strip() for s in e.tags.split(",") if s.strip()],
        priority=e.priority, is_active=e.is_active,
        downloaded=_is_model_downloaded(e.hf_repo, local_path if source != "hf" else None),
        downloading=(e.hf_repo in _catalog_downloading),
        source=source,
        is_public=bool(getattr(e, "is_public", False)),
        local_path=local_path,
        owner_user_id=getattr(e, "owner_user_id", None),
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
    src = req.source if req.source in ("hf", "upload", "finetuned") else "hf"
    entry = ModelCatalogEntry(
        hf_repo=req.hf_repo, params_b=req.params_b, license=req.license,
        hw_min_vram_gb=req.hw_min_vram_gb,
        language=",".join(req.language), tags=",".join(req.tags), priority=req.priority,
        source=src,
        is_public=(src == "hf"),  # HF 는 공개 의미 없음, finetuned/upload 는 비공개 기본
        local_path=req.local_path,
        owner_user_id=actor.id if src in ("upload", "finetuned") else None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _audit(db, actor, "catalog_create", target=entry.id, detail=f"{src}:{req.hf_repo}")
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

    save_state(force=True)
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


# ─────────────────────────────────────────
# 사용자 → 관리자 알림 (PRD §10.5)
# ─────────────────────────────────────────


class NotificationItem(BaseModel):
    id: str
    user_id: str
    user_email: str
    job_id: str | None = None
    kind: str
    error_code: str | None = None
    message: str
    detail: dict
    status: str
    created_at: str
    resolved_at: str | None = None
    resolved_by: str | None = None


def _to_notif(n: AdminNotification) -> NotificationItem:
    return NotificationItem(
        id=n.id, user_id=n.user_id, user_email=n.user_email, job_id=n.job_id,
        kind=n.kind, error_code=n.error_code, message=n.message,
        detail=json.loads(n.detail_json or "{}"),
        status=n.status,
        created_at=n.created_at.isoformat() if n.created_at else "",
        resolved_at=n.resolved_at.isoformat() if n.resolved_at else None,
        resolved_by=n.resolved_by,
    )


@router.get("/notifications", response_model=list[NotificationItem])
def list_notifications(
    status_filter: str = "",
    kind: str = "",
    limit: int = 100,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[NotificationItem]:
    """관리자 알림 목록. status=new/read/resolved, kind=env_missing/hw_mismatch/other 필터."""
    q = db.query(AdminNotification)
    if status_filter:
        q = q.filter(AdminNotification.status == status_filter)
    if kind:
        q = q.filter(AdminNotification.kind == kind)
    rows = q.order_by(AdminNotification.created_at.desc()).limit(limit).all()
    return [_to_notif(n) for n in rows]


class NotificationPatch(BaseModel):
    status: str = Field(pattern="^(new|read|resolved)$")


@router.patch("/notifications/{notif_id}", response_model=NotificationItem)
def patch_notification(
    notif_id: str,
    body: NotificationPatch,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> NotificationItem:
    n = db.get(AdminNotification, notif_id)
    if not n:
        raise HTTPException(status_code=404, detail="Notification not found")
    n.status = body.status
    if body.status == "resolved":
        n.resolved_at = datetime.now(timezone.utc)
        n.resolved_by = actor.id
    db.commit()
    db.refresh(n)
    _audit(db, actor, "notification_patch", target=notif_id, detail=body.status)
    return _to_notif(n)


# ─────────────────────────────────────────
# 기본값 시드 — _ALL_MODELS / PRD §3.2 프리셋 / allow-all 화이트리스트 일괄 삽입
# ─────────────────────────────────────────


class SeedDefaultsResult(BaseModel):
    models_added: int
    presets_added: int
    whitelist_added: int
    models_existing: int
    presets_existing: int
    whitelist_existing: int


# KSIC 별 기본 프리셋 (PRD §3.2). 추천 모델은 _ALL_MODELS 기준 hf_repo 직접 참조.
_PRESET_SEEDS: list[tuple[str, str, str, list[str], str]] = [
    ("C", "제조업 기본", "chat",
     ["Qwen/Qwen2-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
     "제조업 — 매뉴얼/규격 기반 챗"),
    ("G", "도매·소매 기본", "chat",
     ["meta-llama/Llama-3-8B-Instruct", "Qwen/Qwen2-7B-Instruct"],
     "상품 카탈로그/FAQ instruction"),
    ("I", "숙박·음식점 기본", "chat",
     ["google/gemma-2-9b-it", "microsoft/Phi-3-medium-4k-instruct"],
     "메뉴/리뷰 톤·매너 튜닝"),
    ("J", "정보통신·코드", "code",
     ["Qwen/Qwen2.5-Coder-7B-Instruct", "deepseek-ai/deepseek-coder-6.7b-instruct",
      "codellama/CodeLlama-13b-Instruct-hf"],
     "코드 특화"),
    ("K", "금융·보험", "chat",
     ["meta-llama/Llama-3-8B-Instruct", "Qwen/Qwen2-14B-Instruct"],
     "공시/약관 — 안전성 평가 강화"),
    ("M", "전문·과학·기술", "chat",
     ["meta-llama/Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
     "논문/특허 긴 컨텍스트"),
    ("P", "교육", "chat",
     ["google/gemma-2-9b-it", "meta-llama/Llama-3-8B-Instruct"],
     "교과서/문제은행 난이도 조절"),
    ("Q", "보건·의료", "qa",
     ["BioMistral/BioMistral-7B", "epfl-llm/meditron-7b",
      "meta-llama/Llama-3-8B-Instruct"],
     "임상가이드 — 근거성 평가 강화"),
    ("F", "건설", "chat",
     ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2-7B-Instruct"],
     "건설표준시방서/KS"),
]


@router.post("/catalog/seed-defaults", response_model=SeedDefaultsResult)
def seed_defaults(
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> SeedDefaultsResult:
    """
    빈 카탈로그 초기 시드:
      1. _ALL_MODELS → ModelCatalogEntry (hf_repo 중복 건 skip)
      2. PRD §3.2 카테고리 프리셋 (name 중복 skip)
      3. allow-all 화이트리스트 한 건 (없으면 추가)

    같은 hf_repo / preset name / whitelist name 이 이미 있으면 skip.
    soft-delete (is_active=false) 된 행은 그대로 둠 — 의도된 비활성 보존.
    """
    from wfp_worker.stage4_model.model_catalog import _ALL_MODELS

    # 1. 모델
    existing_repos = {e.hf_repo for e in db.query(ModelCatalogEntry).all()}
    models_added = 0
    for m in _ALL_MODELS:
        if m.hf_repo in existing_repos:
            continue
        db.add(ModelCatalogEntry(
            hf_repo=m.hf_repo,
            params_b=m.params_b,
            license=m.license,
            hw_min_vram_gb=m.hw_min_vram_gb,
            language=",".join(m.language),
            tags=",".join(m.tags),
            priority=m.priority,
        ))
        models_added += 1

    # 2. 프리셋
    existing_names = {p.name for p in db.query(CategoryPreset).all()}
    presets_added = 0
    for section, name, task_type, models, desc in _PRESET_SEEDS:
        if name in existing_names:
            continue
        db.add(CategoryPreset(
            name=name,
            ksic_section=section,
            task_type=task_type,
            recommended_models=",".join(models),
            default_config_json="{}",
            description=desc,
        ))
        presets_added += 1

    # 3. allow-all 화이트리스트
    existing_ws = {w.name for w in db.query(DataSourceWhitelist).all()}
    whitelist_added = 0
    if "allow-all" not in existing_ws:
        db.add(DataSourceWhitelist(
            name="allow-all",
            source_type="url",
            url_pattern=".*",
            description="기본 허용 — 모든 URL/도메인 통과. 운영 환경에서는 좁히세요.",
        ))
        whitelist_added += 1

    db.commit()
    _audit(db, actor, "catalog_seed_defaults",
           detail=f"models+{models_added}, presets+{presets_added}, ws+{whitelist_added}")

    return SeedDefaultsResult(
        models_added=models_added,
        presets_added=presets_added,
        whitelist_added=whitelist_added,
        models_existing=len(existing_repos),
        presets_existing=len(existing_names),
        whitelist_existing=len(existing_ws),
    )


# ─────────────────────────────────────────
# 카탈로그 자동 갱신 — HF Hub trending 기반 모델/프리셋 보강
# ─────────────────────────────────────────


class AutoRefreshResult(BaseModel):
    models_added: int
    models_skipped: int
    presets_added: int
    presets_updated: int
    presets_unchanged: int
    hf_search_errors: int
    queries_run: int


@router.post("/catalog/auto-refresh", response_model=AutoRefreshResult)
def auto_refresh_catalog(
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> AutoRefreshResult:
    """
    HF Hub trending + KSIC 섹션 매핑으로 모델/프리셋 자동 갱신.

    - 모델: 라이선스 safelist + 다운로드 임계 통과한 신규 hf_repo 만 INSERT.
      기존 행은 운영자 편집 보존 위해 손대지 않음.
    - 프리셋: 활성 카탈로그 + 섹션 태그 매칭으로 추천 모델 재계산.
      task_type/description 은 비어있을 때만 채움.
    """
    from wfp_worker.admin_tools.catalog_refresher import refresh_catalog

    summary = refresh_catalog(db)
    db.commit()
    _audit(db, actor, "catalog_auto_refresh", detail=str(summary.to_dict()))
    return AutoRefreshResult(**summary.to_dict())


# ─────────────────────────────────────────
# HF Hub 검색 — 카탈로그 추가 시 자동완성 보조 (PRD §10.4 모델 갱신)
# ─────────────────────────────────────────


class HfSearchResultItem(BaseModel):
    hf_repo: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = []
    params_b: float | None = None


# 간단 in-memory 캐시 — (query, limit) → (ts, result). Rate limit 회피.
_hf_cache: dict[tuple[str, int], tuple[float, list[HfSearchResultItem]]] = {}
_HF_CACHE_TTL_SEC = 300  # 5분


def _parse_params_b(model_info: object) -> float | None:
    """HF ModelInfo → 추정 파라미터(B). tags / repo id 의 'XB' 패턴 파싱."""
    import re

    tags = getattr(model_info, "tags", None) or []
    for t in tags:
        m = re.match(r"^(\d+(?:\.\d+)?)b$", str(t).strip().lower())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    repo = getattr(model_info, "id", None) or getattr(model_info, "modelId", None) or ""
    m = re.search(r"(\d+(?:\.\d+)?)b(?:[-_/.]|$)", str(repo).lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


@router.get("/hf-search", response_model=list[HfSearchResultItem])
def hf_search(
    q: str,
    limit: int = 10,
    _: User = Depends(get_current_admin),
) -> list[HfSearchResultItem]:
    """
    HF Hub 모델 검색. 카탈로그 추가 시 hf_repo 자동완성용.

    Parameters
    ----------
    q     검색어 (예: "qwen", "llama-3-8b").
    limit 최대 결과 수 (기본 10, 최대 30).
    """
    q = (q or "").strip()
    if not q:
        return []
    limit = max(1, min(limit, 30))

    cache_key = (q, limit)
    import time
    now = time.time()
    cached = _hf_cache.get(cache_key)
    if cached and (now - cached[0]) < _HF_CACHE_TTL_SEC:
        return cached[1]

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="huggingface_hub 미설치 — 관리자 환경 점검 필요",
        ) from e

    try:
        api = HfApi()
        models = api.list_models(
            search=q, limit=limit, sort="downloads", direction=-1,
        )
        results: list[HfSearchResultItem] = []
        for m in models:
            repo = getattr(m, "id", None) or getattr(m, "modelId", None) or ""
            if not repo:
                continue
            results.append(HfSearchResultItem(
                hf_repo=str(repo),
                downloads=int(getattr(m, "downloads", 0) or 0),
                likes=int(getattr(m, "likes", 0) or 0),
                tags=[str(t) for t in (getattr(m, "tags", None) or [])][:20],
                params_b=_parse_params_b(m),
            ))
    except Exception as e:
        log.warning("hf_search_failed", extra={"extra": {"q": q, "err": str(e)}})
        raise HTTPException(status_code=502, detail=f"HF Hub 검색 실패: {e}") from e

    _hf_cache[cache_key] = (now, results)
    return results


# ─────────────────────────────────────────
# 모델 다운로드 승인 (HF 토큰 / 라이선스 사전 검증 + 재시작)
# ─────────────────────────────────────────


class ModelDownloadResult(BaseModel):
    repo_id: str
    accessible: bool
    status: str   # ok / gated / token_invalid / not_found / error
    detail: str = ""
    hf_url: str = ""
    job_restarted: bool = False


def _hf_preflight(repo_id: str, token: str | None) -> tuple[bool, str, str]:
    """HF 모델 액세스 사전 검증. (accessible, status_code, detail) 반환.

    상태:
      ok            — 토큰 유효 + 라이선스 동의 완료 (다운로드 가능)
      token_invalid — HF_TOKEN 자체 무효 (401)
      gated         — 라이선스 미동의 (403, gated repo)
      not_found     — 리포 미존재 (404)
      error         — 기타 (네트워크 등)
    """
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(repo_id, token=token)
        return True, "ok", "모델 접근 가능 — 다운로드 시작"
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or ("invalid" in msg and "token" in msg):
            return False, "token_invalid", (
                "HF_TOKEN 환경 변수가 유효하지 않음. docker/.env 의 HF_TOKEN 갱신 후 "
                "worker 재시작 (make restart)."
            )
        if "403" in msg or "gated" in msg or "restricted" in msg or "authorized list" in msg:
            return False, "gated", (
                f"라이선스 미동의. https://huggingface.co/{repo_id} 방문 "
                "(HF_TOKEN 발급 계정으로 로그인) → Access repository → 라이선스 동의 "
                "→ 승인 대기 → 본 명령 재시도."
            )
        if "404" in msg or "not found" in msg or "repository not found" in msg:
            return False, "not_found", f"리포 미존재: {repo_id}"
        return False, "error", str(e)[:300]


@router.post("/jobs/{job_id}/model-download", response_model=ModelDownloadResult)
async def admin_model_download(
    job_id: str,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ModelDownloadResult:
    """관리자가 admin notification 의 모델 다운로드 요청을 처리.

    흐름:
      1. job 의 model_hf_repo 추출
      2. HF_TOKEN (env) 으로 사전 검증
         - ok       → Stage 4 부터 파이프라인 재시작
         - gated    → 라이선스 동의 URL 안내 (관리자가 HF 웹 절차 수행 후 재호출)
         - token_invalid / not_found / error → 메시지 반환, 재시작 안 함
    """
    import asyncio as _asyncio

    spec = _jobs.get(job_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="잡 인메모리 없음 — worker 재시작됐을 수 있음")
    repo_id = (spec.model_hf_repo or "").strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="잡에 model_hf_repo 미지정")

    hf_url = f"https://huggingface.co/{repo_id}"
    admin_token = _resolve_admin_hf_token(admin)
    if not admin_token:
        _audit(db, admin, "model_download_request", target=job_id,
               detail=f"token_missing: {repo_id}")
        return ModelDownloadResult(
            repo_id=repo_id,
            accessible=False,
            status="token_missing",
            detail=(
                "HF 토큰 미설정. docker/.env 의 HF_TOKEN 설정 후 worker 재시작, "
                "또는 관리자 설정 화면에서 HF Token 저장 후 재시도."
            ),
            hf_url=hf_url,
            job_restarted=False,
        )

    accessible, s_status, detail = await _asyncio.to_thread(_hf_preflight, repo_id, admin_token)

    # HF 미지원 → 카탈로그 사용불가 마킹
    if s_status == "not_found":
        try:
            entry = db.query(ModelCatalogEntry).filter(
                ModelCatalogEntry.hf_repo == repo_id,
            ).first()
            if entry:
                entry.is_active = False
                db.commit()
                detail = detail + " 카탈로그를 '사용불가' 로 변경했습니다."
        except Exception as e:
            log.warning("catalog_mark_unavailable_failed",
                        extra={"extra": {"err": str(e)}})

    job_restarted = False
    if accessible:
        # admin_token 을 잡 컨텍스트에 주입 (model_downloader._load_token 우선 사용).
        # exclude=True 라 jobs_state.json 영속화 안 됨.
        spec.hf_token = admin_token
        from wfp_worker.runtime import retry_from_stage
        try:
            await retry_from_stage(job_id, 4, reason=f"admin:{admin.email}")
            job_restarted = True
        except Exception as e:
            log.warning("retry_from_stage_failed",
                        extra={"job_id": job_id, "extra": {"err": str(e)}})
            accessible = False
            s_status = "error"
            detail = f"재시작 실패: {e}"

        # 관련 notification 해소 마킹
        try:
            from wfp_worker.db import AdminNotification
            rows = db.query(AdminNotification).filter(
                AdminNotification.job_id == job_id,
                AdminNotification.kind == "model_download",
                AdminNotification.status == "new",
            ).all()
            for r in rows:
                r.status = "resolved"
                r.resolved_at = datetime.now(timezone.utc)
                r.resolved_by = admin.id
            db.commit()
        except Exception as e:
            log.warning("notification_resolve_failed",
                        extra={"extra": {"err": str(e)}})

    _audit(db, admin, "model_download_request", target=job_id,
           detail=f"{s_status}: {repo_id}")

    return ModelDownloadResult(
        repo_id=repo_id,
        accessible=accessible,
        status=s_status,
        detail=detail,
        hf_url=hf_url,
        job_restarted=job_restarted,
    )


# ─────────────────────────────────────────
# 카탈로그 모델 다운로드 (admin 트리거)
# ─────────────────────────────────────────


class EnvHfTokenStatus(BaseModel):
    has_env_token: bool   # docker/.env 의 HF_TOKEN 설정 여부
    has_admin_db_token: bool   # 현재 admin 의 DB hf_token_encrypted 설정 여부
    effective_source: str   # "env" / "admin_db" / "none"


@router.get("/env/hf-token-status", response_model=EnvHfTokenStatus)
def env_hf_token_status(
    admin: User = Depends(get_current_admin),
) -> EnvHfTokenStatus:
    """HF 토큰 가용 경로 확인. Frontend 가 다운로드 가능 여부 판정에 사용."""
    import os as _os
    has_env = bool(_os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    has_db = bool(getattr(admin, "hf_token_encrypted", None))
    if has_env:
        src = "env"
    elif has_db:
        src = "admin_db"
    else:
        src = "none"
    return EnvHfTokenStatus(has_env_token=has_env, has_admin_db_token=has_db, effective_source=src)


def _resolve_admin_hf_token(admin: User) -> str | None:
    """Admin 컨텍스트에서 사용할 HF 토큰. env > admin DB > None."""
    import os as _os
    return (
        _os.environ.get("HF_TOKEN")
        or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or getattr(admin, "hf_token_encrypted", None)
        or None
    )


class CatalogDownloadResult(BaseModel):
    hf_repo: str
    status: str   # ok / gated / token_invalid / token_missing / not_found / already_downloaded / downloading / error
    detail: str = ""
    hf_url: str = ""


def _do_catalog_download(repo_id: str, token: str | None) -> None:
    """백그라운드 다운로드 worker. 완료 시 _catalog_downloading 에서 제거."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        log.info("catalog_download_done", extra={"extra": {"repo": repo_id}})
    except Exception as e:
        log.warning("catalog_download_failed",
                    extra={"extra": {"repo": repo_id, "err": str(e)[:200]}})
        _catalog_downloading[repo_id] = {**_catalog_downloading.get(repo_id, {}), "error": str(e)[:300]}
        return
    finally:
        _catalog_downloading.pop(repo_id, None)


@router.post("/catalog/models/{entry_id}/download", response_model=CatalogDownloadResult)
async def catalog_download(
    entry_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> CatalogDownloadResult:
    """카탈로그 모델 다운로드 트리거. HF pre-flight → 백그라운드 다운로드 시작.

    HF 측이 모델을 미지원(404)으로 응답하면 catalog.is_active=False 마킹 → '사용불가'.
    """
    import asyncio as _asyncio
    entry = db.get(ModelCatalogEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")

    repo_id = entry.hf_repo
    hf_url = f"https://huggingface.co/{repo_id}"

    # 이미 cache 에 있으면 skip
    if _is_model_downloaded(repo_id):
        return CatalogDownloadResult(
            hf_repo=repo_id, status="already_downloaded",
            detail="이미 로컬 HF cache 에 존재", hf_url=hf_url,
        )

    # 진행 중 중복 방지
    if repo_id in _catalog_downloading:
        return CatalogDownloadResult(
            hf_repo=repo_id, status="downloading",
            detail="이미 다운로드 진행 중", hf_url=hf_url,
        )

    token = _resolve_admin_hf_token(admin)
    if not token:
        return CatalogDownloadResult(
            hf_repo=repo_id, status="token_missing",
            detail=(
                "HF 토큰 미설정. docker/.env 의 HF_TOKEN 설정 후 worker 재시작, "
                "또는 관리자 설정 화면에서 HF Token 저장."
            ),
            hf_url=hf_url,
        )

    accessible, s_status, detail = await _asyncio.to_thread(_hf_preflight, repo_id, token)

    if s_status == "not_found":
        # HF 미지원 — 카탈로그에서 사용불가 마킹
        entry.is_active = False
        db.commit()
        _audit(db, admin, "catalog_mark_unavailable", target=entry_id, detail=repo_id)
        return CatalogDownloadResult(
            hf_repo=repo_id, status="not_found",
            detail=detail + " 카탈로그를 '사용불가' 로 변경했습니다.",
            hf_url=hf_url,
        )

    if not accessible:
        return CatalogDownloadResult(
            hf_repo=repo_id, status=s_status, detail=detail, hf_url=hf_url,
        )

    # 백그라운드 다운로드 시작
    import time as _time
    _catalog_downloading[repo_id] = {"started_at": _time.time(), "error": None}
    _asyncio.create_task(_asyncio.to_thread(_do_catalog_download, repo_id, token))
    _audit(db, admin, "catalog_download_start", target=entry_id, detail=repo_id)

    return CatalogDownloadResult(
        hf_repo=repo_id, status="ok",
        detail="다운로드 시작 — 카탈로그 새로고침으로 진행 확인",
        hf_url=hf_url,
    )


# ─────────────────────────────────────────
# 카탈로그 공개/비공개 토글 + 사용자 모델 파일 업로드
# ─────────────────────────────────────────


class TogglePublicResult(BaseModel):
    id: str
    hf_repo: str
    source: str
    is_public: bool


@router.post("/catalog/models/{entry_id}/toggle-public", response_model=TogglePublicResult)
def catalog_toggle_public(
    entry_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> TogglePublicResult:
    """사용자(비공개)↔(공개) 토글. source='hf' 항목에는 의미 없음."""
    entry = db.get(ModelCatalogEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    src = getattr(entry, "source", "hf") or "hf"
    if src == "hf":
        raise HTTPException(
            status_code=400,
            detail="HF 공식 모델은 공개/비공개 토글 불가 (source='hf')",
        )
    entry.is_public = not bool(getattr(entry, "is_public", False))
    db.commit()
    _audit(db, actor, "catalog_toggle_public", target=entry_id,
           detail=f"is_public={entry.is_public}")
    return TogglePublicResult(
        id=entry.id, hf_repo=entry.hf_repo, source=src,
        is_public=bool(entry.is_public),
    )


# 업로드 모델 저장 루트 — ~/.wfp/uploads/<entry_id>/
_UPLOAD_ROOT = Path.home() / ".wfp" / "uploads"


@router.post("/catalog/models/upload", response_model=CatalogItem, status_code=status.HTTP_201_CREATED)
async def catalog_upload_model(
    hf_repo: str = Form(...),
    params_b: float = Form(...),
    license: str = Form("custom"),
    hw_min_vram_gb: float = Form(0.0),
    language: str = Form("ko,en"),
    tags: str = Form("upload"),
    priority: int = Form(50),
    is_public: bool = Form(False),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_admin),
) -> CatalogItem:
    """사용자 모델 파일 업로드 + 카탈로그 등록.

    multipart/form-data — 한 모델당 다수 파일 (config.json, *.safetensors, tokenizer.* 등).
    저장 경로: ~/.wfp/uploads/<entry_id>/
    hf_repo 식별: 충돌 방지 위해 'upload://' 접두 강제 (사용자 자유 입력시).
    """
    import uuid as _uuid

    # 입력 정리
    repo = hf_repo.strip()
    if not repo:
        raise HTTPException(status_code=400, detail="hf_repo 필수")
    if not repo.startswith("upload://"):
        repo = f"upload://{repo.lstrip('/')}"

    if db.query(ModelCatalogEntry).filter(ModelCatalogEntry.hf_repo == repo).first():
        raise HTTPException(status_code=409, detail=f"Already exists: {repo}")

    if not files:
        raise HTTPException(status_code=400, detail="파일 1 개 이상 필요")

    entry_id = str(_uuid.uuid4())
    dest_dir = _UPLOAD_ROOT / entry_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 파일 저장 (chunked)
    saved = []
    total_bytes = 0
    for uf in files:
        if not uf.filename:
            continue
        # path traversal 차단
        safe_name = uf.filename.replace("..", "_").replace("/", "_").replace("\\", "_")
        path = dest_dir / safe_name
        size = 0
        with path.open("wb") as fp:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                fp.write(chunk)
                size += len(chunk)
        saved.append({"name": safe_name, "size": size})
        total_bytes += size

    if not saved:
        # 빈 업로드 — 디렉토리 정리
        try:
            import shutil as _sh
            _sh.rmtree(dest_dir, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="유효한 파일 없음")

    entry = ModelCatalogEntry(
        id=entry_id,
        hf_repo=repo,
        params_b=params_b,
        license=license,
        hw_min_vram_gb=hw_min_vram_gb,
        language=language,
        tags=tags,
        priority=priority,
        is_active=True,
        source="upload",
        is_public=is_public,
        local_path=str(dest_dir),
        owner_user_id=actor.id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    _audit(db, actor, "catalog_upload", target=entry.id,
           detail=f"{repo} files={len(saved)} bytes={total_bytes}")
    log.info("catalog_upload_done",
             extra={"extra": {"id": entry.id, "files": len(saved), "bytes": total_bytes}})

    return _to_catalog_item(entry)
