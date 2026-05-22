"""
E2E API 테스트 — ml-frontend → ml-worker 흐름 통합 검증.

FastAPI TestClient(=Starlette httpx)로 실제 서버 라우팅 + DB 사용.
in-memory SQLite로 격리. 각 테스트마다 DB 새로 생성.

시나리오:
  E1   /health 동작
  E2   /hw/caps 비인증 호출 가능
  E3   signup → 즉시 access_token 반환
  E4   첫 가입 사용자 자동 ADMIN
  E5   두 번째 사용자는 USER
  E6   중복 이메일 signup → 409
  E7   login 정상
  E8   login 잘못된 비번 → 401
  E9   /auth/me 인증 → UserPublic
  E10  /auth/me 토큰 없으면 401
  E11  /users/me PATCH 정상
  E12  /users/me/hf-token 저장/삭제
  E13  /models/ 비인증 OK + 후보 반환
  E14  /jobs GET — 인증 필요
  E15  /jobs POST — 잡 생성 + 인메모리 등록
  E16  /jobs/{id} — 본인 잡 조회 가능, 타인은 403
  E17  /jobs/{id} DELETE — 본인만
  E18  /admin/users — ADMIN만
  E19  /admin/users PATCH — 권한 변경 + audit 기록
  E20  /admin/catalog/models CRUD + audit
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 테스트 전용 DB URL을 import 전에 환경 변수로 주입 — 세션 모듈이 이 값을 읽음
# 임시 sqlite 파일 사용 (in-memory는 multi-connection 이슈)
_TMP_DB_FILE = Path(tempfile.gettempdir()) / "wfp_e2e_test.db"
if _TMP_DB_FILE.exists():
    _TMP_DB_FILE.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB_FILE}"
os.environ["JWT_SECRET"] = "test-e2e-secret"

from wfp_worker.db import Base, get_engine  # noqa: E402
from wfp_worker.server import app  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db() -> Iterator[None]:
    """각 테스트 시작 전 DB 테이블 drop+create."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    # 인메모리 잡 스토어도 초기화
    from wfp_worker.runtime import _jobs, _ws_clients
    _jobs.clear()
    _ws_clients.clear()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _signup(client: TestClient, email: str, password: str = "secret123") -> str:
    """헬퍼 — signup → access_token 반환."""
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_admin(email: str) -> None:
    """DB에서 직접 role=ADMIN 설정 (테스트 전용 — SUPERVISOR 없이 ADMIN 필요한 케이스)."""
    from wfp_worker.db import db_session
    from wfp_worker.db.models import User, UserRole
    with db_session() as db:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.role = UserRole.ADMIN


# ─────────────────────────────────────────
# 시스템
# ─────────────────────────────────────────


def test_e1_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_e2_hw_caps_unauthed(client: TestClient) -> None:
    r = client.get("/hw/caps")
    assert r.status_code == 200
    data = r.json()
    assert "os" in data and "ram_gb" in data


# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────


def test_e3_signup_returns_token(client: TestClient) -> None:
    r = client.post("/auth/signup", json={"email": "alice@example.com", "password": "secret123"})
    assert r.status_code == 201
    body = r.json()
    assert body["access_token"]
    assert body["user"]["email"] == "alice@example.com"


def test_e4_first_user_is_user(client: TestClient) -> None:
    """회원가입은 항상 USER — ADMIN은 SUPERVISOR 환경변수 또는 관리자 직접 부여."""
    r = client.post("/auth/signup", json={"email": "first@example.com", "password": "secret123"})
    assert r.json()["user"]["role"] == "USER"


def test_e5_second_user_is_user(client: TestClient) -> None:
    client.post("/auth/signup", json={"email": "admin@x.com", "password": "secret123"})
    r = client.post("/auth/signup", json={"email": "bob@x.com", "password": "secret123"})
    assert r.json()["user"]["role"] == "USER"


def test_e6_duplicate_email(client: TestClient) -> None:
    client.post("/auth/signup", json={"email": "dup@x.com", "password": "secret123"})
    r = client.post("/auth/signup", json={"email": "dup@x.com", "password": "secret456"})
    assert r.status_code == 409


def test_e7_login_ok(client: TestClient) -> None:
    client.post("/auth/signup", json={"email": "login@x.com", "password": "secret123"})
    r = client.post("/auth/login", json={"email": "login@x.com", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_e8_login_bad_password(client: TestClient) -> None:
    client.post("/auth/signup", json={"email": "bad@x.com", "password": "secret123"})
    r = client.post("/auth/login", json={"email": "bad@x.com", "password": "WRONG"})
    assert r.status_code == 401


def test_e9_me_with_token(client: TestClient) -> None:
    token = _signup(client, "me@x.com")
    r = client.get("/auth/me", headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["email"] == "me@x.com"


def test_e10_me_without_token(client: TestClient) -> None:
    r = client.get("/auth/me")
    assert r.status_code == 401


# ─────────────────────────────────────────
# Users
# ─────────────────────────────────────────


def test_e11_patch_profile(client: TestClient) -> None:
    token = _signup(client, "patch@x.com")
    r = client.patch("/users/me", headers=_auth(token), json={"name": "Wiseman"})
    assert r.status_code == 200
    assert r.json()["name"] == "Wiseman"


def test_e12_hf_token_save_delete(client: TestClient) -> None:
    token = _signup(client, "hf@x.com")
    r = client.post("/users/me/hf-token", headers=_auth(token), json={"hf_token": "hf_xxxxxx12345"})
    assert r.status_code == 204
    r = client.delete("/users/me/hf-token", headers=_auth(token))
    assert r.status_code == 204


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────


def test_e13_models_unauthed(client: TestClient) -> None:
    r = client.get("/models/?section=J&max_vram_gb=16&language=ko")
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list)
    assert len(items) > 0
    # J 섹션 → code 태그 모델 존재 확인
    assert any("code" in m["tags"] for m in items)


# ─────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────


def test_e14_jobs_requires_auth(client: TestClient) -> None:
    r = client.get("/jobs")
    assert r.status_code == 401


def test_e15_create_job(client: TestClient) -> None:
    token = _signup(client, "job@x.com")
    body = {
        "ksic_section": "J",
        "task_type": "code",
        "description": "테스트용 잡 — 충분히 긴 설명입니다.",
    }
    r = client.post("/jobs", headers=_auth(token), json=body)
    assert r.status_code == 202, r.text
    assert "job_id" in r.json()


def test_e16_get_job_ownership(client: TestClient) -> None:
    token_a = _signup(client, "owner@x.com")
    token_b = _signup(client, "stranger@x.com")
    body = {"ksic_section": "J", "task_type": "chat", "description": "owner 의 잡입니다 충분히 길게."}
    r = client.post("/jobs", headers=_auth(token_a), json=body)
    job_id = r.json()["job_id"]

    # 본인 조회 OK
    r2 = client.get(f"/jobs/{job_id}", headers=_auth(token_a))
    assert r2.status_code == 200

    # 타인은 403
    r3 = client.get(f"/jobs/{job_id}", headers=_auth(token_b))
    assert r3.status_code == 403


def test_e17_cancel_job_ownership(client: TestClient) -> None:
    token_a = _signup(client, "owner2@x.com")
    token_b = _signup(client, "stranger2@x.com")
    r = client.post("/jobs", headers=_auth(token_a), json={
        "ksic_section": "J", "task_type": "chat",
        "description": "취소 테스트 충분히 긴 설명입니다.",
    })
    job_id = r.json()["job_id"]

    # 타인 취소 시도 → 403
    r1 = client.delete(f"/jobs/{job_id}", headers=_auth(token_b))
    assert r1.status_code == 403

    # 본인 취소 OK
    r2 = client.delete(f"/jobs/{job_id}", headers=_auth(token_a))
    assert r2.status_code in (200, 409)  # 이미 다른 상태일 수 있음


# ─────────────────────────────────────────
# Admin
# ─────────────────────────────────────────


def test_e18_admin_only(client: TestClient) -> None:
    # 첫 가입자가 ADMIN, 두 번째가 USER
    _signup(client, "boss@x.com")
    user_token = _signup(client, "peon@x.com")
    r = client.get("/admin/users", headers=_auth(user_token))
    assert r.status_code == 403


def test_e19_admin_patch_user_with_audit(client: TestClient) -> None:
    admin_token = _signup(client, "boss@x.com")
    _make_admin("boss@x.com")
    _signup(client, "peon@x.com")

    # 사용자 목록에서 peon ID 추출
    users = client.get("/admin/users", headers=_auth(admin_token)).json()
    peon = next(u for u in users if u["email"] == "peon@x.com")

    # 권한 변경
    r = client.patch(
        f"/admin/users/{peon['id']}", headers=_auth(admin_token),
        json={"role": "ADMIN"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "ADMIN"

    # audit 기록 확인
    audit = client.get("/admin/audit", headers=_auth(admin_token)).json()
    assert any(a["action"] == "user_patch" for a in audit)


def test_e20_catalog_crud(client: TestClient) -> None:
    admin = _signup(client, "boss@x.com")
    _make_admin("boss@x.com")

    # 생성
    r = client.post("/admin/catalog/models", headers=_auth(admin), json={
        "hf_repo": "test/custom-model",
        "params_b": 7.0,
        "license": "apache-2.0",
        "hw_min_vram_gb": 6.0,
        "language": ["ko"],
        "tags": ["custom"],
    })
    assert r.status_code == 201
    entry_id = r.json()["id"]

    # 조회
    r = client.get("/admin/catalog/models", headers=_auth(admin))
    assert any(e["id"] == entry_id for e in r.json())

    # /models/ 공개 엔드포인트에도 노출
    r = client.get("/models/?section=&max_vram_gb=24")
    assert any(m["hf_repo"] == "test/custom-model" for m in r.json())

    # 수정
    r = client.patch(f"/admin/catalog/models/{entry_id}", headers=_auth(admin),
                     json={"priority": 1})
    assert r.json()["priority"] == 1

    # 삭제 (soft)
    r = client.delete(f"/admin/catalog/models/{entry_id}", headers=_auth(admin))
    assert r.status_code == 204

    # audit 기록
    audit = client.get("/admin/audit", headers=_auth(admin)).json()
    actions = {a["action"] for a in audit}
    assert {"catalog_create", "catalog_patch", "catalog_delete"} <= actions
