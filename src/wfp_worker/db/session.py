"""
DB 엔진 / 세션 / 초기화.

환경 변수 `DATABASE_URL` 기준:
  - 미설정 또는 sqlite://...  → SQLite (dev)
  - postgresql://...           → PostgreSQL (prod)

ml-worker 컨테이너에서:
  - dev: sqlite:////data/wfp.db (compose volume)
  - prod: postgresql://wfp:...@postgres:5432/wfp
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from wfp_worker.common.logger import get_logger

from .base import Base

log = get_logger(__name__)


def _default_sqlite_url() -> str:
    """dev 기본 — ~/.wfp/wfp.db."""
    db_dir = Path.home() / ".wfp"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir / 'wfp.db'}"


_DATABASE_URL = os.environ.get("DATABASE_URL", _default_sqlite_url())

# SQLite는 같은 스레드 외 접근 금지가 기본 — FastAPI multi-thread 위해 해제.
_engine_kwargs: dict = {"pool_pre_ping": True}
if _DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

_engine: Engine = create_engine(_DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_engine() -> Engine:
    """전역 엔진 반환 (테스트에서 override 시 사용)."""
    return _engine


def init_db() -> None:
    """
    스타트업 시 호출 — 테이블 생성 + 슈퍼바이저 계정 보장.

    프로덕션에서는 Alembic 마이그레이션 권장. 데모/개발 편의용.
    """
    # 모델 등록 (import 부수 효과)
    from . import models  # noqa: F401

    log.info("DB 초기화", extra={"extra": {"url": _DATABASE_URL.split("@")[-1]}})
    Base.metadata.create_all(bind=_engine)
    _migrate_add_columns()
    ensure_supervisor()


def _migrate_add_columns() -> None:
    """경량 마이그레이션 — 신규 컬럼 누락 시 ALTER TABLE ADD COLUMN.

    Alembic 미사용 시 schema drift 회피. SQLite + PostgreSQL 양립.
    추가 컬럼 정의: (table, column, type, default_sql).
    """
    from sqlalchemy import text
    additions = [
        ("model_catalog", "source", "VARCHAR(16)", "'hf'"),
        ("model_catalog", "is_public", "BOOLEAN", "0"),
        ("model_catalog", "local_path", "VARCHAR(512)", "NULL"),
        ("model_catalog", "owner_user_id", "VARCHAR(36)", "NULL"),
    ]
    with _engine.begin() as conn:
        for table, col, col_type, default_sql in additions:
            try:
                if _DATABASE_URL.startswith("sqlite"):
                    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    existing = {r[1] for r in rows}
                else:
                    rows = conn.execute(text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = :t"
                    ), {"t": table}).fetchall()
                    existing = {r[0] for r in rows}
                if col in existing:
                    continue
                default_clause = (
                    f" DEFAULT {default_sql}" if default_sql and default_sql != "NULL" else ""
                )
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}{default_clause}"
                ))
                log.info("schema_migrate_add_column",
                         extra={"extra": {"table": table, "col": col}})
            except Exception as e:
                log.warning("schema_migrate_failed",
                            extra={"extra": {"table": table, "col": col, "err": str(e)}})


def ensure_supervisor() -> None:
    """
    SUPERVISOR_ID / SUPERVISOR_PASSWORD 환경변수로 관리자 계정 보장.

    - 미존재 시 ADMIN 계정 생성.
    - 존재 시 ADMIN 권한 유지 + 비밀번호 env 값으로 동기화.
    - env 미설정 시 스킵 (기존 계정 영향 없음).
    """
    supervisor_email = os.environ.get("SUPERVISOR_ID", "").strip()
    supervisor_pw = os.environ.get("SUPERVISOR_PASSWORD", "").strip()

    if not supervisor_email or not supervisor_pw:
        return

    from wfp_worker.auth.security import hash_password

    from .models import User, UserRole

    with db_session() as db:
        user = db.query(User).filter(User.email == supervisor_email).first()
        if user is None:
            db.add(User(
                email=supervisor_email,
                password_hash=hash_password(supervisor_pw),
                role=UserRole.ADMIN,
                name="Supervisor",
                is_active=True,
            ))
            log.info("supervisor_created", extra={"extra": {"email": supervisor_email}})
        else:
            user.role = UserRole.ADMIN
            user.password_hash = hash_password(supervisor_pw)
            log.info("supervisor_ensured", extra={"extra": {"email": supervisor_email}})


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI Depends 용 세션 제공자.

    사용:
        @router.get("/me")
        def me(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """일반 컨텍스트 매니저 — 라우터 외부 (CLI, 백그라운드)에서 사용."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
