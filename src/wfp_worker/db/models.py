"""
ORM 모델.

PRD §8.2 핵심 엔티티 중 pipeline 외 부분만 DB에 저장:
  - users         (회원, 권한)
  - audit_logs    (관리자 액션)
  - model_catalog (관리자가 추가/수정하는 모델 카탈로그)

jobs/artifacts는 현재 인메모리 (실시간 진행 상태). 향후 DB 영속화 가능.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_str() -> str:
    return str(uuid.uuid4())


class UserRole(str, enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class User(Base):
    """회원 + 권한."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 안전 저장된 HF 토큰 (개인 키체인 의도)
    hf_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    """관리자 액션 감사 로그 (PRD §10.5)."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    actor_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    actor_email: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)


class ModelCatalogEntry(Base):
    """
    관리자가 추가/수정하는 모델 카탈로그.

    기존 stage4_model.model_catalog의 _ALL_MODELS는 시드 데이터.
    운영 시 이 테이블이 정본이 된다.
    """

    __tablename__ = "model_catalog"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    hf_repo: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    params_b: Mapped[float] = mapped_column(Float, nullable=False)
    license: Mapped[str] = mapped_column(String(64), nullable=False)
    hw_min_vram_gb: Mapped[float] = mapped_column(Float, nullable=False)
    language: Mapped[str] = mapped_column(String(255), default="ko,en")   # CSV
    tags: Mapped[str] = mapped_column(String(255), default="general")     # CSV
    priority: Mapped[int] = mapped_column(Integer, default=5)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )
