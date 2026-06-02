"""
FastAPI ML Worker 서버.

라우터 마운트 + WebSocket(`/ws/{job_id}`) + 헬스체크.
세부 API는 `wfp_worker.api/` 패키지의 라우터에서 처리.

엔드포인트:
  GET  /health           헬스체크
  GET  /hw/caps          현재 HW 스냅샷
  WS   /ws/{job_id}      실시간 진행 스트림 (Bearer 토큰 query param 옵션)

  /auth/*    회원가입 / 로그인 / me
  /users/*   프로필 / HF 토큰
  /models/*  모델 카탈로그
  /jobs/*    잡 CRUD + 아티팩트 다운로드
  /admin/*   관리자
"""

from __future__ import annotations

# .env 파일 로드 — 다른 모듈 import 전에 실행해야 module-level 환경변수에 반영됨
from dotenv import load_dotenv
load_dotenv()

# tqdm → WS 브리지 설치. transformers/datasets/huggingface_hub 등 외부 라이브러리
# import 이전에 호출해야 그들이 보는 tqdm 클래스가 패치됨.
from wfp_worker.common import tqdm_bridge as _tqdm_bridge  # noqa: E402
_tqdm_bridge.install()

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from wfp_worker.api import (
    admin_router,
    auth_router,
    data_router,
    jobs_router,
    models_router,
    users_router,
)
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobState
from wfp_worker.db import init_db
from wfp_worker.runtime import _jobs, _ws_clients, get_job_events

log = get_logger(__name__, service="wfp-worker-api")


# ─────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):
    log.info("WFP Worker 시작 — DB 초기화")
    init_db()
    # 디스크에서 이전 잡 + 이벤트 복원 (컨테이너 재시작 후 이력 보존)
    from wfp_worker.runtime import load_state, save_state
    load_state()
    yield
    # 종료 직전 한번 더 저장 (graceful shutdown) — throttle 우회
    save_state(force=True)
    log.info("WFP Worker 종료")


# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────

app = FastAPI(
    title="WFP ML Worker",
    version="0.1.0",
    description="Wise Fine-tuning Pipeline ML Worker API",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 마운트
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(models_router)
app.include_router(data_router)
app.include_router(jobs_router)
app.include_router(admin_router)


# ─────────────────────────────────────────
# 시스템 엔드포인트
# ─────────────────────────────────────────


@app.get("/health", tags=["system"])
async def health() -> dict:
    """헬스체크."""
    return {"status": "ok", "jobs": len(_jobs)}


@app.get("/hw/caps", tags=["system"])
async def hw_caps() -> dict:
    """현재 HW 스냅샷 (인증 불필요 — 회원가입 전 dashboard 미리보기 용)."""
    from wfp_worker.stage2_hw_probe import HWProber
    caps = await asyncio.to_thread(HWProber().probe)
    return caps.model_dump()


# ─────────────────────────────────────────
# WebSocket — 진행 스트림
# ─────────────────────────────────────────


@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str) -> None:
    """
    실시간 진행 스트림.

    NOTE: WS는 헤더 인증이 복잡하므로 현재 인증 우회.
    운영에서는 `?token=...` query param 또는 첫 메시지 인증 권장.
    """
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    log.info("ws_connect_attempt", extra={"job_id": job_id, "extra": {"client": client, "in_jobs": job_id in _jobs}})
    if job_id not in _jobs:
        log.warning("ws_reject_unknown_job", extra={"job_id": job_id, "extra": {"known_jobs": len(_jobs)}})
        await websocket.close(code=4004)
        return

    try:
        await websocket.accept()
    except Exception as e:
        log.exception("ws_accept_failed", extra={"job_id": job_id, "extra": {"err": str(e)}})
        return
    log.info("ws_accepted", extra={"job_id": job_id})
    _ws_clients.setdefault(job_id, []).append(websocket)

    try:
        spec = _jobs[job_id]
        # 1. 현재 상태 메타 — backfill 시작 표시
        await websocket.send_text(json.dumps({
            "job_id": job_id,
            "msg": "connected",
            "extra": {
                "state": spec.state.value,
                "stage": spec.current_stage,
                "backfill_start": True,
            },
        }))

        # 2. 누적된 이벤트 backfill (잡이 이미 진행됐거나 종료됐어도 전체 로그 복원)
        for past in get_job_events(job_id):
            await websocket.send_text(past)

        # 3. backfill 끝 표시 — 클라이언트는 이후부터 라이브 스트림
        await websocket.send_text(json.dumps({
            "job_id": job_id,
            "msg": "backfill_end",
            "extra": {"state": spec.state.value, "stage": spec.current_stage},
        }))

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text('{"ping": true}')
            except WebSocketDisconnect:
                break

            if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
                break
    finally:
        clients = _ws_clients.get(job_id, [])
        if websocket in clients:
            clients.remove(websocket)


# ─────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
