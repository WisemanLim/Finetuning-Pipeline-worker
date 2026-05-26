"""
비밀번호 해시 + JWT 토큰 생성/검증.

Argon2id 해시 (PRD §9 인증/권한).
JWT는 HS256 — 운영에서는 RS256 + KMS 권장.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

# Argon2id 컨텍스트
_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# JWT 비밀키 — 운영에서는 반드시 환경 변수로 교체.
# os.environ.get("K", default)은 K가 빈 문자열이면 빈 문자열 반환 → `or` 체인으로
# 빈 값도 fallback. JWT_SECRET → NEXTAUTH_SECRET → dev default 순.
_JWT_SECRET = (
    os.environ.get("JWT_SECRET")
    or os.environ.get("NEXTAUTH_SECRET")
    or "dev-jwt-secret-CHANGE-ME"
)
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MIN = int(os.environ.get("JWT_ACCESS_EXPIRE_MIN", "10080"))  # 7 days


def hash_password(plain: str) -> str:
    """Argon2id 해시."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """비밀번호 검증 (constant time)."""
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception:
        return False


def create_access_token(
    subject: str,
    role: str = "USER",
    expire_min: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    JWT 액세스 토큰 생성.

    Parameters
    ----------
    subject:
        sub 클레임 — 보통 user_id.
    role:
        권한 (USER / ADMIN).
    expire_min:
        만료(분). 미지정 시 환경변수 또는 60분.
    extra:
        추가 클레임 (email 등).
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_min or _ACCESS_TOKEN_EXPIRE_MIN)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    JWT 검증 + payload 반환.

    Raises
    ------
    jwt.InvalidTokenError
        만료, 서명 불일치, 형식 오류 등.
    """
    return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
