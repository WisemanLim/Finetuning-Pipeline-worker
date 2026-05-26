"""
런타임 공유 상태 + 파이프라인 실행기.

server.py와 api/jobs.py 양쪽에서 import하는 단일 모듈.
순환 임포트 방지 + 인메모리 잡 스토어 공유.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any, Deque

from fastapi import WebSocket

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import DataSourceMode, JobSpec, JobState

log = get_logger(__name__, service="wfp-worker-runtime")

# ─────────────────────────────────────────
# 인메모리 상태
# ─────────────────────────────────────────

# 잡 ID → JobSpec (실시간 상태)
_jobs: dict[str, JobSpec] = {}

# 잡 ID → 연결된 WS 클라이언트
_ws_clients: dict[str, list[WebSocket]] = {}

# 잡 ID → 이벤트 히스토리 (WS 재연결/사후 조회 시 backfill용)
# 잡당 최근 500개 유지 (메모리 폭주 방지).
_EVENT_BUFFER_SIZE = 500
_job_events: dict[str, Deque[str]] = {}


def get_job_events(job_id: str) -> list[str]:
    """잡의 누적된 이벤트 JSON 문자열 목록 반환."""
    buf = _job_events.get(job_id)
    return list(buf) if buf else []


# ─────────────────────────────────────────
# 영속화 — 컨테이너 재시작에도 잡 + 이벤트 buffer 보존
# ─────────────────────────────────────────

# 우선순위: /data (compose volume) > ~/.wfp (로컬 dev)
_PERSIST_DIR = Path("/data") if Path("/data").exists() else Path.home() / ".wfp"
_PERSIST_PATH = _PERSIST_DIR / "jobs_state.json"


_BACKUP_KEEP = 5  # 최근 N개 회전 백업 유지


def _rotate_backup() -> None:
    """덮어쓰기 전 jobs_state.json 회전 백업.

    실수 삭제 / 손상 시 복구 가능. jobs_state.json.1 ~ jobs_state.json.N 형태.
    """
    if not _PERSIST_PATH.exists():
        return
    try:
        # N → N+1 (가장 오래된 것 삭제)
        oldest = _PERSIST_PATH.with_suffix(f".json.{_BACKUP_KEEP}")
        if oldest.exists():
            oldest.unlink()
        for i in range(_BACKUP_KEEP - 1, 0, -1):
            src = _PERSIST_PATH.with_suffix(f".json.{i}")
            if src.exists():
                src.rename(_PERSIST_PATH.with_suffix(f".json.{i + 1}"))
        # 현재 → .1
        import shutil
        shutil.copy2(_PERSIST_PATH, _PERSIST_PATH.with_suffix(".json.1"))
    except Exception as e:
        log.debug("backup rotate 실패", extra={"extra": {"err": str(e)}})


def save_state() -> None:
    """잡 메타데이터 + 이벤트 buffer를 디스크에 atomic write + 회전 백업."""
    try:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        # 덮어쓰기 전 이전 버전 백업 (실수 삭제 복구용)
        _rotate_backup()
        payload = {
            "jobs": {jid: spec.model_dump(mode="json") for jid, spec in _jobs.items()},
            "events": {jid: list(buf) for jid, buf in _job_events.items()},
        }
        tmp = _PERSIST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(_PERSIST_PATH)  # atomic rename
    except Exception as e:
        log.warning("save_state 실패", extra={"extra": {"err": str(e), "path": str(_PERSIST_PATH)}})


def load_state() -> None:
    """디스크에서 잡 + 이벤트 복원. 서버 lifespan startup에서 호출."""
    if not _PERSIST_PATH.exists():
        log.info("영속화 파일 없음 — 빈 상태로 시작",
                 extra={"extra": {"path": str(_PERSIST_PATH)}})
        return

    try:
        data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
        for jid, spec_data in data.get("jobs", {}).items():
            try:
                _jobs[jid] = JobSpec.model_validate(spec_data)
            except Exception as e:
                log.warning("잡 복원 실패 — 스키마 불일치", extra={"extra": {"jid": jid, "err": str(e)}})

        for jid, ev_list in data.get("events", {}).items():
            buf: Deque[str] = deque(maxlen=_EVENT_BUFFER_SIZE)
            for ev in ev_list:
                buf.append(ev)
            _job_events[jid] = buf

        log.info(
            f"영속화 복원 완료 — 잡 {len(_jobs)}건",
            extra={"extra": {"path": str(_PERSIST_PATH), "jobs": len(_jobs), "events": len(_job_events)}},
        )
    except Exception as e:
        log.warning("load_state 실패 — 빈 상태로 시작",
                    extra={"extra": {"err": str(e), "path": str(_PERSIST_PATH)}})


# ─────────────────────────────────────────
# WS 이벤트 발행
# ─────────────────────────────────────────


async def _emit_ws(job_id: str, msg: str, extra: dict | None = None) -> None:
    """
    잡에 연결된 모든 WS 클라이언트에 메시지 전송 + 이벤트 buffer 누적.

    buffer는 클라이언트가 나중에 (잡 진행 중 또는 완료 후) WS에 연결할 때
    이전 이벤트를 backfill하기 위해 사용.
    """
    payload = json.dumps({"job_id": job_id, "msg": msg, "extra": extra or {}}, ensure_ascii=False)

    # buffer 누적 (WS 클라이언트 유무와 무관)
    buf = _job_events.setdefault(job_id, deque(maxlen=_EVENT_BUFFER_SIZE))
    buf.append(payload)

    # 디스크 영속화 — 컨테이너 재시작에도 이력 보존
    save_state()

    # 라이브 전송
    clients = _ws_clients.get(job_id, [])
    if not clients:
        return
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


def _make_progress_cb(job_id: str):
    """동기 콜백 → asyncio 이벤트 브리지 (Stage 7 trainer 콜백용)."""
    def cb(jid: str, stage: int, msg: str, extra: dict) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_emit_ws(jid, msg, {**extra, "stage": stage}))
        except RuntimeError:
            pass
    return cb


# ─────────────────────────────────────────
# 파이프라인 실행기 (9-Stage)
# ─────────────────────────────────────────


async def _run_pipeline(spec: JobSpec, req: dict[str, Any]) -> None:
    """
    9-Stage 파이프라인 전체 실행.

    req는 JobCreateRequest.model_dump() 결과 dict.
    """
    job_id = str(spec.job_id)

    async def emit(msg: str, extra: dict | None = None) -> None:
        await _emit_ws(job_id, msg, extra)

    from datetime import datetime, timezone
    spec.started_at = datetime.now(timezone.utc)

    try:
        # Stage 2 — HW Probe
        await emit("Stage 2: HW 프로빙 시작")
        from wfp_worker.stage2_hw_probe import HWProber, PlanRecommender
        caps = await asyncio.to_thread(HWProber().probe)
        spec.hw_caps = caps
        recommender = PlanRecommender()
        try:
            plan = await asyncio.to_thread(recommender.recommend, caps)
        except WFPError as e:
            if e.code == WFPErrorCode.NO_PLAN_MATCH and req.get("allow_force_plan"):
                # 사용자 동의로 강제 진행 — fallback plan 생성 + 경고 broadcast
                await emit(
                    f"⚠ [{e.code.value}] {e.message} — 사용자 동의로 강제 진행",
                    {"stage": 2, "warning": True, "force_plan": True, "detail": e.detail},
                )
                plan = await asyncio.to_thread(recommender.force_fallback_plan, caps)
                await emit(
                    f"강제 fallback plan 적용: {plan.backend} / max {plan.max_model_params_b}B / {plan.recommended_strategy}",
                    {"stage": 2, "warning": True, "force_plan": True,
                     "max_params_b": plan.max_model_params_b, "backend": plan.backend},
                )
            else:
                # detail 정보를 WS event에도 노출 (사용자 confirmation 화면용)
                await emit(
                    f"[{e.code.value}] {e.message}",
                    {"stage": 2, "code": e.code.value, "detail": e.detail,
                     "can_force": e.code == WFPErrorCode.NO_PLAN_MATCH},
                )
                raise
        spec.recommended_plan = plan
        await emit("Stage 2 완료", {"max_vram_gb": caps.max_vram_gb, "stage": 2})

        # Stage 3 — Env Setup
        await emit("Stage 3: 환경 검사")
        from wfp_worker.stage3_env_setup import EnvManager
        env_result = await asyncio.to_thread(EnvManager().setup, spec)
        if env_result.driver_warnings:
            await emit("환경 경고", {"warnings": env_result.driver_warnings, "stage": 3})

        # Stage 4 — Model
        await emit("Stage 4: 모델 선택", {"stage": 4})
        from wfp_worker.stage4_model import ModelDownloader, ModelPicker
        await asyncio.to_thread(ModelPicker().pick, spec, req.get("model_hf_repo") or None)
        await emit("Stage 4: 모델 다운로드 중", {"repo": spec.model_hf_repo, "stage": 4})
        await asyncio.to_thread(ModelDownloader().download, spec)
        await emit("Stage 4 완료", {"model": spec.model_hf_repo, "stage": 4})

        # Stage 5 — Data
        await emit("Stage 5: 데이터 수집 시작", {"stage": 5})
        from wfp_worker.stage5_data import DataCleaner, DataCollector, DataFormatter

        collector = DataCollector()
        raw_path = await asyncio.to_thread(
            collector.collect,
            spec,
            DataSourceMode(req.get("data_source_mode", "upload")),
            req.get("data_source_path", ""),
            req.get("keywords", []),
        )
        raw_records = []
        with Path(raw_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        cleaned, stats = await asyncio.to_thread(DataCleaner().clean, raw_records)
        await emit("Stage 5: 정제 완료", {"records": stats.output_count, "stage": 5})

        await asyncio.to_thread(DataFormatter().format_and_split, spec, cleaned)
        await emit("Stage 5 완료", {"train": spec.num_train, "val": spec.num_val, "stage": 5})

        # Stage 6
        await emit("Stage 6: 학습 설정 생성", {"stage": 6})
        from wfp_worker.stage6_train_config import TrainConfigBuilder
        config = await asyncio.to_thread(
            TrainConfigBuilder().build, spec, req.get("train_override") or None,
        )
        await emit("Stage 6 완료", {"strategy": config.strategy.value, "stage": 6})

        # Stage 7
        await emit("Stage 7: 학습 시작", {"stage": 7})
        from wfp_worker.stage7_train import PipelineTrainer
        cb = _make_progress_cb(job_id)
        await asyncio.to_thread(PipelineTrainer(progress_cb=cb).train, spec)
        await emit("Stage 7 완료", {"loss": spec.train_loss, "stage": 7})

        # Stage 8
        await emit("Stage 8: 평가 시작", {"stage": 8})
        from wfp_worker.stage8_eval import ModelEvaluator
        result = await asyncio.to_thread(ModelEvaluator().evaluate, spec)
        await emit("Stage 8 완료", {"passed": result.passed, "stage": 8})

        # Stage 9
        await emit("Stage 9: 패키징 시작", {"stage": 9})
        from wfp_worker.stage9_package import Packager
        artifacts = await asyncio.to_thread(Packager().package, spec)
        await emit("파이프라인 완료", {
            "artifacts": [a.kind.value for a in artifacts], "stage": 9,
        })

        spec.state = JobState.DONE
        spec.finished_at = datetime.now(timezone.utc)

    except WFPError as e:
        spec.state = JobState.FAILED
        spec.error_code = e.code.value
        spec.error_message = e.message
        spec.finished_at = datetime.now(timezone.utc)
        await emit("오류 발생", {"code": e.code.value, "msg": e.message})
        log.error("pipeline_failed", extra={"job_id": job_id, "extra": e.to_dict()})

    except Exception as e:
        spec.state = JobState.FAILED
        spec.error_message = str(e)
        spec.finished_at = datetime.now(timezone.utc)
        await emit("예기치 못한 오류", {"err": str(e)})
        log.exception("pipeline_crashed", extra={"job_id": job_id})
