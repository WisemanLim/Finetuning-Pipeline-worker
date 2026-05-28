"""
tqdm → WebSocket 브리지.

transformers / datasets / huggingface_hub 등 외부 라이브러리가 stdout 으로
출력하는 tqdm 진행도를 잡 WebSocket 이벤트로 자동 발행한다.

설계:
- `WfpTqdm` 은 `tqdm.std.tqdm` 서브클래스. `update`/`refresh`/`close` 호출 시
  500ms throttle 로 WS emit.
- `install()` 호출 시 `tqdm.std.tqdm` 과 `tqdm.auto.tqdm` 을 교체.
  반드시 transformers/datasets import 이전에 한 번만 호출.
- 잡 컨텍스트는 `contextvars.ContextVar` 로 전달. `asyncio.to_thread` 는
  Python 3.9+ 부터 호출 시 컨텍스트를 자동 복사 → 워커 스레드에서도 유효.
- emit 콜백은 `configure(emitter)` 로 등록. 메인 루프와의 thread-safe 디스패치는
  runtime._make_progress_cb 와 동일 패턴.

부작용 차단:
- emitter 미설정 / job_id 미설정 / total<=0 인 경우 NO-OP.
- emit 실패는 학습/평가에 영향 주지 않도록 silent.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Callable

# job_id 컨텍스트 — 잡 진행 중에만 set.
_active_job_id: ContextVar[str | None] = ContextVar(
    "_wfp_tqdm_job_id", default=None,
)
# stage 컨텍스트 — 현재 stage 번호 (1~9).
_active_stage: ContextVar[int] = ContextVar(
    "_wfp_tqdm_stage", default=0,
)

# 외부에서 등록하는 emit 콜백. signature: (job_id, stage, msg, extra) -> None.
# runtime._make_progress_cb 와 호환되도록 동기 함수로 받음.
_EmitFn = Callable[[str, int, str, dict[str, Any]], None]
_emitter: _EmitFn | None = None

# 설치됨 플래그 — 이중 install 방지.
_installed: bool = False


def configure(emitter: _EmitFn) -> None:
    """emit 콜백 등록. 한 번만 호출. runtime 모듈에서 main loop 캡처 후 호출."""
    global _emitter
    _emitter = emitter


def set_job_context(job_id: str, stage: int) -> tuple[Any, Any]:
    """현재 task 의 잡 컨텍스트 설정. 반환된 토큰을 reset 에 사용."""
    return (_active_job_id.set(job_id), _active_stage.set(stage))


def reset_job_context(tokens: tuple[Any, Any]) -> None:
    """set_job_context 가 반환한 토큰으로 복원."""
    try:
        _active_job_id.reset(tokens[0])
        _active_stage.reset(tokens[1])
    except (ValueError, LookupError):
        pass


def install() -> bool:
    """tqdm.std.tqdm + tqdm.auto.tqdm 을 WfpTqdm 으로 교체.

    Returns
    -------
    True   교체 성공.
    False  이미 설치됨 / tqdm 미설치 등.
    """
    global _installed
    if _installed:
        return False
    try:
        import tqdm as tqdm_mod  # type: ignore
        import tqdm.auto as tqdm_auto  # type: ignore
        import tqdm.std as tqdm_std  # type: ignore
    except ImportError:
        return False

    _Original = tqdm_std.tqdm  # noqa: N806

    class WfpTqdm(_Original):  # type: ignore[misc, valid-type]
        """update/refresh 시 WS emit. throttle 500ms."""

        _wfp_last_emit_at: float = 0.0
        _wfp_emit_interval: float = 0.5

        # tqdm.__init__ 에서 인스턴스 속성 _wfp_last_emit_at 을 명시적으로 초기화
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._wfp_last_emit_at = 0.0

        def update(self, n: float = 1) -> Any:
            result = super().update(n)
            try:
                self._wfp_maybe_emit()
            except Exception:
                pass
            return result

        def refresh(self, *args: Any, **kwargs: Any) -> Any:
            result = super().refresh(*args, **kwargs)
            try:
                self._wfp_maybe_emit()
            except Exception:
                pass
            return result

        def close(self) -> Any:
            try:
                self._wfp_emit(force=True)
            except Exception:
                pass
            return super().close()

        def _wfp_maybe_emit(self) -> None:
            # disable 인스턴스는 _wfp_last_emit_at 도 없을 수 있음 → getattr 폴백.
            last = getattr(self, "_wfp_last_emit_at", 0.0)
            now = time.monotonic()
            if now - last < self._wfp_emit_interval:
                return
            self._wfp_last_emit_at = now
            self._wfp_emit()

        def _wfp_emit(self, force: bool = False) -> None:
            if _emitter is None:
                return
            job_id = _active_job_id.get()
            if not job_id:
                return
            # tqdm(disable=True) 인스턴스는 fast-path __init__ 으로 일부 속성
            # (desc, dynamic_ncols 등) 이 설정되지 않음 → 직접 self.desc 접근 시
            # AttributeError: 'tqdm' object has no attribute 'desc'.
            # getattr 로 안전하게 폴백.
            if getattr(self, "disable", False):
                return
            total = getattr(self, "total", None) or 0
            if total <= 0:
                return
            n = getattr(self, "n", 0) or 0
            desc = getattr(self, "desc", None) or "progress"
            stage = _active_stage.get()
            try:
                _emitter(
                    job_id, stage,
                    f"{desc}: {n}/{total}",
                    {
                        "step": int(n),
                        "total_steps": int(total),
                        "tqdm_desc": desc,
                        "stage_progress": (n / total) if total else 0.0,
                    },
                )
            except Exception:
                pass

    # tqdm.std.tqdm 과 tqdm.auto.tqdm 둘 다 교체.
    tqdm_std.tqdm = WfpTqdm
    tqdm_auto.tqdm = WfpTqdm
    # 일부 라이브러리는 `from tqdm import tqdm` 으로 import → top-level 도 교체.
    tqdm_mod.tqdm = WfpTqdm

    _installed = True
    return True
