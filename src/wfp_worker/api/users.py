"""
사용자 자기 자신 관리 라우터.

엔드포인트:
  PATCH /users/me              프로필 수정 (이름 등)

DEPRECATED (2026-05): /users/me/hf-token 저장/삭제는 비활성. gated repo 다운로드는
관리자 알림(AdminNotification kind=hf_gated) → 관리자가 HF_TOKEN 환경변수로
download 수행하는 흐름으로 대체. 잠재적 평문 토큰 누수 + UX 혼선 회피 목적.
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
    # 최대 길이 제거 — HF 토큰 포맷 변경 / 회사 wrapper 토큰 대비.
    # 양끝 공백은 서버에서 strip 후 검증.
    hf_token: str = Field(min_length=4, max_length=4096)


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


# DEPRECATED 2026-05: 사용자 HF 토큰 저장/삭제 라우트 주석 처리.
# gated repo 접근은 관리자 알림 → 관리자 토큰 다운로드 흐름으로 대체.
# 기존 hf_token_encrypted 컬럼은 스키마 호환 위해 유지 (마이그레이션 회피),
# 신규 저장은 차단. 필요 시 재활성화는 본 블록 주석 해제 + admin.py 의
# resume-download 흐름과 정책 정합 확인 후 진행.
#
# @router.post("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
# def save_hf_token(
#     req: HfTokenRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ) -> None:
#     """HF 토큰 저장. 평문 저장 — 운영 KMS/Vault 권장."""
#     token = (req.hf_token or "").strip()
#     if not token:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="HF token 이 비어있습니다.",
#         )
#     user.hf_token_encrypted = token
#     db.commit()
#
#
# @router.delete("/me/hf-token", status_code=status.HTTP_204_NO_CONTENT)
# def delete_hf_token(
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ) -> None:
#     """저장된 HF 토큰 제거."""
#     user.hf_token_encrypted = None
#     db.commit()
