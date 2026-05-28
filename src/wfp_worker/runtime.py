"""
런타임 공유 상태 + 파이프라인 실행기.

server.py와 api/jobs.py 양쪽에서 import하는 단일 모듈.
순환 임포트 방지 + 인메모리 잡 스토어 공유.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque

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

# 잡 ID → 현재 실행 중인 파이프라인 asyncio.Task (advance/finalize 시 취소용)
_job_tasks: dict[str, asyncio.Task] = {}

# 잡 ID → 생성 시점의 JobCreateRequest dict (advance 시 stage 재실행용)
_job_reqs: dict[str, dict[str, Any]] = {}

# 잡 ID → 마지막 WS 이벤트 ISO timestamp (stuck 감지용)
_job_last_event_at: dict[str, str] = {}

# 잡 ID → advance/finalize 직렬화용 Lock
_advance_locks: dict[str, asyncio.Lock] = {}


def get_job_events(job_id: str) -> list[str]:
    """잡의 누적된 이벤트 JSON 문자열 목록 반환."""
    buf = _job_events.get(job_id)
    return list(buf) if buf else []


def get_last_event_at(job_id: str) -> str | None:
    """마지막 이벤트 timestamp (ISO8601). 없으면 None."""
    return _job_last_event_at.get(job_id)


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
            "reqs": _job_reqs.copy(),
            "last_event_at": _job_last_event_at.copy(),
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

        # advance/finalize 지원용 부가 메타 — 신규 필드, 구버전 dump 호환
        for jid, req in data.get("reqs", {}).items():
            if isinstance(req, dict):
                _job_reqs[jid] = req
        for jid, ts in data.get("last_event_at", {}).items():
            if isinstance(ts, str):
                _job_last_event_at[jid] = ts

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

    # 마지막 이벤트 timestamp 갱신 — stuck 감지에 사용
    _job_last_event_at[job_id] = datetime.now(timezone.utc).isoformat()

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
    """
    동기 콜백 → asyncio 이벤트 브리지 (Stage 7 trainer 콜백용).

    cb 는 `asyncio.to_thread(trainer.train, ...)` 안의 워커 스레드에서 호출되므로
    `asyncio.get_event_loop()` / `ensure_future` 는 동작하지 않는다.
    이 함수가 async 컨텍스트에서 호출될 때 메인 루프 참조를 캡처해
    `run_coroutine_threadsafe` 로 안전하게 디스패치한다.
    """
    main_loop = asyncio.get_running_loop()

    def cb(jid: str, stage: int, msg: str, extra: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                _emit_ws(jid, msg, {**extra, "stage": stage}),
                main_loop,
            )
        except Exception:
            # 루프가 이미 닫혔거나 디스패치 실패 — 학습은 계속
            pass

    # tqdm_bridge 가 사용할 동일 emit 콜백 등록 (idempotent).
    # transformers / datasets / huggingface_hub 의 tqdm 진행도가 WS 로 흐름.
    try:
        from wfp_worker.common import tqdm_bridge
        tqdm_bridge.configure(cb)
    except Exception:
        pass

    return cb


# ─────────────────────────────────────────
# 파이프라인 실행기 (9-Stage) — stage 별 분해
#
# 각 _stage_N(spec, req, emit) 함수는 한 stage 의 작업만 담당.
# _run_pipeline 이 start_stage 부터 9 까지 순회 호출.
# advance/finalize 가 코루틴 task 를 취소하고 start_stage 를 바꿔 재spawn.
# ─────────────────────────────────────────


_EmitFn = Callable[..., Awaitable[None]]


async def _stage_2_hw(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 2 — HW Probe + Plan 결정."""
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
            await emit(
                f"[{e.code.value}] {e.message}",
                {"stage": 2, "code": e.code.value, "detail": e.detail,
                 "can_force": e.code == WFPErrorCode.NO_PLAN_MATCH},
            )
            raise
    spec.recommended_plan = plan
    await emit("Stage 2 완료", {"max_vram_gb": caps.max_vram_gb, "stage": 2})


async def _stage_3_env(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 3 — Env Setup."""
    await emit("Stage 3: 환경 검사")
    from wfp_worker.stage3_env_setup import EnvManager
    env_result = await asyncio.to_thread(EnvManager().setup, spec)
    if env_result.driver_warnings:
        await emit("환경 경고", {"warnings": env_result.driver_warnings, "stage": 3})


async def _stage_4_model(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 4 — Model Pick + Download."""
    await emit("Stage 4: 모델 선택", {"stage": 4})
    from wfp_worker.stage4_model import ModelDownloader, ModelPicker
    await asyncio.to_thread(ModelPicker().pick, spec, req.get("model_hf_repo") or None)
    await emit("Stage 4: 모델 다운로드 중", {"repo": spec.model_hf_repo, "stage": 4})
    await asyncio.to_thread(ModelDownloader().download, spec)
    await emit("Stage 4 완료", {"model": spec.model_hf_repo, "stage": 4})


async def _stage_5_data(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 5 — Data Collect / Clean / Format."""
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


async def _stage_6_config(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 6 — Train Config 생성."""
    await emit("Stage 6: 학습 설정 생성", {"stage": 6})
    from wfp_worker.stage6_train_config import TrainConfigBuilder
    config = await asyncio.to_thread(
        TrainConfigBuilder().build, spec, req.get("train_override") or None,
    )
    await emit("Stage 6 완료", {"strategy": config.strategy.value, "stage": 6})


async def _stage_7_train(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 7 — 학습 실행."""
    await emit("Stage 7: 학습 시작", {"stage": 7})
    from wfp_worker.stage7_train import PipelineTrainer
    cb = _make_progress_cb(str(spec.job_id))
    await asyncio.to_thread(PipelineTrainer(progress_cb=cb).train, spec)
    await emit("Stage 7 완료", {"loss": spec.train_loss, "stage": 7})


async def _stage_8_eval(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 8 — 평가."""
    await emit("Stage 8: 평가 시작", {"stage": 8})
    from wfp_worker.stage8_eval import ModelEvaluator
    eval_cb = _make_progress_cb(str(spec.job_id))
    result = await asyncio.to_thread(ModelEvaluator().evaluate, spec, eval_cb)
    await emit("Stage 8 완료", {"passed": result.passed, "stage": 8})


async def _stage_9_package(spec: JobSpec, req: dict[str, Any], emit: _EmitFn) -> None:
    """Stage 9 — 패키징."""
    await emit("Stage 9: 패키징 시작", {"stage": 9})
    from wfp_worker.stage9_package import Packager
    artifacts = await asyncio.to_thread(Packager().package, spec)
    await emit("파이프라인 완료", {
        "artifacts": [a.kind.value for a in artifacts], "stage": 9,
    })


_STAGE_RUNNERS: dict[int, Callable[..., Awaitable[None]]] = {
    2: _stage_2_hw,
    3: _stage_3_env,
    4: _stage_4_model,
    5: _stage_5_data,
    6: _stage_6_config,
    7: _stage_7_train,
    8: _stage_8_eval,
    9: _stage_9_package,
}


async def _run_pipeline(
    spec: JobSpec,
    req: dict[str, Any],
    *,
    start_stage: int = 2,
) -> None:
    """
    9-Stage 파이프라인 실행. start_stage 부터 9 까지 순회.

    Parameters
    ----------
    spec:
        잡 JobSpec (in-memory 인스턴스).
    req:
        JobCreateRequest.model_dump() — stage 별 입력 dict.
    start_stage:
        시작 stage 번호 (기본 2 = 첫 실행). advance 시 N+1 로 전달.
    """
    job_id = str(spec.job_id)

    async def emit(msg: str, extra: dict | None = None) -> None:
        await _emit_ws(job_id, msg, extra)

    if start_stage <= 2:
        spec.started_at = datetime.now(timezone.utc)

    # tqdm_bridge: 첫 호출 시 emit cb 등록 + 매 stage 마다 context 설정.
    # transformers/datasets 의 stdout tqdm 도 WS 이벤트로 흐르게 됨.
    from wfp_worker.common import tqdm_bridge
    _bridge_cb = _make_progress_cb(job_id)  # configure 부수 효과
    tqdm_bridge.configure(_bridge_cb)

    try:
        for stage_num in range(start_stage, 10):
            if stage_num not in _STAGE_RUNNERS:
                continue
            spec.current_stage = stage_num
            runner = _STAGE_RUNNERS[stage_num]
            ctx_tokens = tqdm_bridge.set_job_context(job_id, stage_num)
            try:
                await runner(spec, req, emit)
            finally:
                tqdm_bridge.reset_job_context(ctx_tokens)

        spec.state = JobState.DONE
        spec.finished_at = datetime.now(timezone.utc)

    except asyncio.CancelledError:
        # advance/finalize 가 호출 — 새 task 가 이어받으므로 깔끔히 종료
        log.info("pipeline_cancelled",
                 extra={"job_id": job_id, "extra": {"at_stage": spec.current_stage}})
        raise

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


# ─────────────────────────────────────────
# Advance / Finalize — stuck 잡 수동 진행
# ─────────────────────────────────────────


def _free_accelerator_memory() -> None:
    """GPU/MPS 캐시 해제. 취소된 stage 가 점유한 메모리 회수."""
    try:
        import torch  # type: ignore
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():  # type: ignore
            try:
                torch.mps.empty_cache()  # type: ignore
            except Exception:
                pass
    except Exception:
        # torch 미설치 / import 실패 → skip
        pass


async def _await_quietly(task: asyncio.Task) -> None:
    """Task 종료를 기다리되 모든 예외 무시."""
    try:
        await task
    except BaseException:  # noqa: BLE001 — cancel/일반 예외 모두 무시
        pass


async def _cancel_running_task(job_id: str, timeout: float = 5.0) -> bool:
    """현재 실행 중인 pipeline task 를 취소하고 종료를 기다림.

    Returns
    -------
    True  — task 존재하여 취소 시도함 (이미 종료된 경우 포함).
    False — task 없음.
    """
    task = _job_tasks.get(job_id)
    if task is None:
        return False
    if task.done():
        return True
    task.cancel()
    try:
        await asyncio.wait_for(_await_quietly(task), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("pipeline_task_cancel_timeout",
                    extra={"job_id": job_id, "extra": {"timeout_sec": timeout}})
    _free_accelerator_memory()
    return True


def _get_lock(job_id: str) -> asyncio.Lock:
    """advance/finalize 직렬화용 잡 별 Lock 반환."""
    lock = _advance_locks.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        _advance_locks[job_id] = lock
    return lock


async def advance_job(job_id: str, *, reason: str = "user") -> dict[str, Any]:
    """
    현재 stage 강제 종료 + 다음 stage 부터 재실행.

    Returns
    -------
    {advanced_from, new_start_stage, restarted}
    """
    async with _get_lock(job_id):
        spec = _jobs.get(job_id)
        if spec is None:
            raise WFPError(WFPErrorCode.JOB_NOT_FOUND, f"job {job_id} 없음")

        if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
            raise WFPError(
                WFPErrorCode.INVALID_STATE,
                f"종료된 잡은 advance 불가 (state={spec.state.value})",
            )

        from_stage = spec.current_stage
        next_stage = from_stage + 1

        if next_stage > 9:
            # 이미 마지막 stage — finalize 처리
            await _emit_ws(job_id, "마지막 stage 도달 — 강제 완료 처리",
                           {"stage": from_stage, "advance": True, "reason": reason})
            spec.state = JobState.DONE
            spec.finished_at = datetime.now(timezone.utc)
            await _cancel_running_task(job_id)
            return {"advanced_from": from_stage, "new_start_stage": None, "restarted": False}

        # 현재 task 취소
        await _cancel_running_task(job_id)

        await _emit_ws(
            job_id,
            f"Stage {from_stage} 강제 스킵 → Stage {next_stage} 부터 재개",
            {"stage": from_stage, "advance": True, "reason": reason,
             "from_stage": from_stage, "to_stage": next_stage},
        )

        req = _job_reqs.get(job_id) or {}
        spec.current_stage = next_stage - 1  # 루프 시작 시 next_stage 로 set
        spec.state = JobState.RUNNING
        new_task = asyncio.create_task(_run_pipeline(spec, req, start_stage=next_stage))
        _job_tasks[job_id] = new_task

        log.info(
            "job_advanced",
            extra={
                "job_id": job_id,
                "extra": {"from": from_stage, "to": next_stage, "reason": reason},
            },
        )
        return {"advanced_from": from_stage, "new_start_stage": next_stage, "restarted": True}


async def finalize_job(job_id: str, *, reason: str = "user") -> dict[str, Any]:
    """
    학습(Stage 7) 산출물로 잡 조기 종료. 평가/패키징 스킵.

    체크포인트 (`spec.checkpoint_dir`) 존재해야 의미 있음.
    Stage 9 만 강제 실행해 model_card / zip artifact 만 생성하는 옵션도 가능하나,
    MVP 는 DONE 마킹 + 산출물은 사용자가 직접 다운로드.
    """
    async with _get_lock(job_id):
        spec = _jobs.get(job_id)
        if spec is None:
            raise WFPError(WFPErrorCode.JOB_NOT_FOUND, f"job {job_id} 없음")

        if spec.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
            raise WFPError(
                WFPErrorCode.INVALID_STATE,
                f"이미 종료된 잡 (state={spec.state.value})",
            )

        # 현재 task 취소
        await _cancel_running_task(job_id)

        await _emit_ws(
            job_id, "잡 강제 종료 (finalize)",
            {"stage": spec.current_stage, "finalize": True, "reason": reason,
             "checkpoint": spec.checkpoint_dir},
        )

        spec.state = JobState.DONE
        spec.finished_at = datetime.now(timezone.utc)
        save_state()

        log.info(
            "job_finalized",
            extra={"job_id": job_id, "extra": {"at_stage": spec.current_stage, "reason": reason}},
        )
        return {
            "finalized_at_stage": spec.current_stage,
            "checkpoint_dir": spec.checkpoint_dir,
            "state": spec.state.value,
        }
