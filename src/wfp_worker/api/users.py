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


# 일반 사용자 영역에서는 HF 토큰 입력 미노출 (settings 페이지에서 admin role 일 때만 렌더).
# 백엔드 endpoint 는 admin 권한 검증 — 잘못된 호출 시 403.
class HfTokenRequest(BaseModel):
    hf_token: str = Field(min_length=10, max_length=255)


class UserDetail(BaseModel):
    id: str
    email: str
    name: str | None
    role: str
    has_hf_token: bool = False  # admin 의 DB hf_token_encrypted 또는 env HF_TOKEN 존재 여부


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
    # has_hf_token: admin 본인 DB 값 또는 env HF_TOKEN 존재 여부
    import os as _os
    env_token = bool(_os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    db_token = bool(getattr(user, "hf_token_encrypted", None))
    return UserDetail(
        id=user.id, email=user.email, name=user.name,
        role=user.role.value,
        has_hf_token=(env_token or db_token),
    )


# HF 토큰 저장/삭제 — admin role 전용 (UI 도 admin role 일때만 노출).
# Worker 가 모델 다운로드 시 우선순위: spec.hf_token > env HF_TOKEN > admin DB.
@router.post("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
def save_hf_token(
    req: HfTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """HF 토큰 저장 (admin 만). 형식 간단 검증."""
    if user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="HF 토큰 저장은 관리자 권한 필요",
        )
    if not req.hf_token.startswith(("hf_", "api_")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HF token must start with 'hf_' or 'api_'",
        )
    user.hf_token_encrypted = req.hf_token  # TODO: encrypt (Fernet / vault)
    db.commit()


@router.delete("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
def delete_hf_token(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """저장된 HF 토큰 제거 (admin 만)."""
    if user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="HF 토큰 삭제는 관리자 권한 필요",
        )
    user.hf_token_encrypted = None
    db.commit()
