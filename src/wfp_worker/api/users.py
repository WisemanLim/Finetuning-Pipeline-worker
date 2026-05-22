"""
사용자 자기 자신 관리 라우터.

엔드포인트:
  PATCH /users/me              프로필 수정 (이름 등)
  POST  /users/me/hf-token     HuggingFace 토큰 저장 (서버 측 암호화)
  DELETE /users/me/hf-token    HF 토큰 삭제
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from wfp_worker.auth import get_current_user
from wfp_worker.db import User, get_db

router = APIRouter(prefix="/users", tags=["users"])


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=128)


class HfTokenRequest(BaseModel):
    hf_token: str = Field(min_length=10, max_length=255)


class UserDetail(BaseModel):
    id: str
    email: str
    name: str | None
    role: str
    has_hf_token: bool


@router.patch("/me", response_model=UserDetail)
def update_profile(
    req: ProfileUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserDetail:
    """이름 변경 등 기본 프로필 갱신."""
    if req.name is not None:
        user.name = req.name
    db.commit()
    return UserDetail(
        id=user.id, email=user.email, name=user.name,
        role=user.role.value, has_hf_token=bool(user.hf_token_encrypted),
    )


@router.post("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
def save_hf_token(
    req: HfTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """
    HF 토큰 저장.

    운영에서는 KMS / Vault로 암호화 후 저장 권장.
    본 데모는 평문 저장 (TODO: Fernet 또는 vault 연동).
    """
    # 형식 간단 검증 ("hf_" 접두 권장)
    if not req.hf_token.startswith(("hf_", "api_")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HF token must start with 'hf_' or 'api_'",
        )
    user.hf_token_encrypted = req.hf_token  # TODO: encrypt
    db.commit()


@router.delete("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
def delete_hf_token(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """저장된 HF 토큰 제거."""
    user.hf_token_encrypted = None
    db.commit()
