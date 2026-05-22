"""
인증 라우터.

엔드포인트:
  POST /auth/signup    회원가입 (이메일 + 비밀번호)
  POST /auth/login     로그인 → JWT access 토큰
  GET  /auth/me        현재 사용자 정보 (Bearer 토큰 필요)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from wfp_worker.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from wfp_worker.common.logger import get_logger
from wfp_worker.db import User, UserRole, get_db

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ─────────────────────────────────────────
# 요청/응답 스키마
# ─────────────────────────────────────────


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

    user: "UserPublic"


class UserPublic(BaseModel):
    id: str
    email: str
    name: str | None = None
    role: str
    is_active: bool

    @classmethod
    def from_orm(cls, u: User) -> "UserPublic":
        return cls(id=u.id, email=u.email, name=u.name, role=u.role.value, is_active=u.is_active)


TokenResponse.model_rebuild()


# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """회원가입 + 즉시 로그인 토큰 발급."""
    # 중복 이메일 검사
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    role = UserRole.USER

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        name=req.name,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log.info("signup", extra={"user_id": user.id, "extra": {"email": req.email, "role": role.value}})

    token = create_access_token(subject=user.id, role=user.role.value, extra={"email": user.email})
    return TokenResponse(access_token=token, user=UserPublic.from_orm(user))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """이메일+비밀번호 로그인."""
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(subject=user.id, role=user.role.value, extra={"email": user.email})
    log.info("login", extra={"user_id": user.id})
    return TokenResponse(access_token=token, user=UserPublic.from_orm(user))


@router.get("/me", response_model=UserPublic)
def me(user: User = Depends(get_current_user)) -> UserPublic:
    """현재 세션 사용자."""
    return UserPublic.from_orm(user)
