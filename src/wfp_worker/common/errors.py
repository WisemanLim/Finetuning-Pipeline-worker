"""
WFP 에러 코드 및 예외 클래스.

에러 코드 체계: WFP-XXYY
  XX = 스테이지 번호 (01~09)
  YY = 에러 일련번호

예시:
  WFP-0101 = Stage 1, 에러 01 (잘못된 KSIC 섹션)
  WFP-0401 = Stage 4, 에러 01 (HF 다운로드 실패)
"""

from enum import Enum


class WFPErrorCode(str, Enum):
    """파이프라인 전역 에러 코드."""

    # Stage 1 — Init
    INVALID_KSIC_SECTION = "WFP-0101"
    INVALID_TASK_TYPE = "WFP-0102"
    MISSING_DESCRIPTION = "WFP-0103"

    # Stage 2 — HW Probe
    HW_PROBE_FAILED = "WFP-0201"
    NVML_UNAVAILABLE = "WFP-0202"
    NO_PLAN_MATCH = "WFP-0203"

    # Stage 3 — Env Setup
    PYTHON_VERSION_MISMATCH = "WFP-0301"
    MISSING_DEPENDENCY = "WFP-0302"
    DRIVER_NOT_FOUND = "WFP-0303"
    UV_NOT_FOUND = "WFP-0304"

    # Stage 4 — Model
    HF_DOWNLOAD_FAILED = "WFP-0401"
    MODEL_NOT_FOUND = "WFP-0402"
    LICENSE_NOT_ACCEPTED = "WFP-0403"
    INSUFFICIENT_DISK = "WFP-0404"
    HF_TOKEN_MISSING = "WFP-0405"

    # Stage 5 — Data
    DATA_UPLOAD_FAILED = "WFP-0501"
    DATA_DIR_NOT_FOUND = "WFP-0502"
    DATA_TOO_SMALL = "WFP-0503"
    DATA_FORMAT_ERROR = "WFP-0504"
    DATA_CLEAN_FAILED = "WFP-0505"

    # Stage 6 — Train Config
    CONFIG_BUILD_FAILED = "WFP-0601"
    STRATEGY_NOT_SUPPORTED = "WFP-0602"

    # Stage 7 — Train
    TRAIN_CUDA_OOM = "WFP-0701"
    TRAIN_CRASHED = "WFP-0702"
    CHECKPOINT_SAVE_FAILED = "WFP-0703"

    # Stage 8 — Eval
    EVAL_FAILED = "WFP-0801"
    EVAL_METRIC_MISSING = "WFP-0802"
    EVAL_QUALITY_BELOW_THRESHOLD = "WFP-0803"

    # Stage 9 — Package
    GGUF_CONVERT_FAILED = "WFP-0901"
    ARTIFACT_UPLOAD_FAILED = "WFP-0902"
    MODEL_CARD_FAILED = "WFP-0903"

    # 공통 / 운영
    JOB_NOT_FOUND = "WFP-9001"          # advance/finalize 대상 잡 없음
    INVALID_STATE = "WFP-9002"          # 종료된 잡에 진행 시도 등 상태 위반
    UNKNOWN = "WFP-9999"


class WFPError(Exception):
    """WFP 파이프라인 전용 예외.

    모든 파이프라인 오류는 이 클래스로 래핑하여 일관된 에러 코드와
    메시지를 보장한다.
    """

    def __init__(
        self,
        code: WFPErrorCode,
        message: str,
        stage: int | None = None,
        detail: str = "",
    ) -> None:
        self.code = code
        self.message = message
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> dict:
        """RFC 7807 스타일 에러 페이로드."""
        return {
            "type": f"urn:wfp:error:{self.code.value}",
            "title": self.message,
            "code": self.code.value,
            "stage": self.stage,
            "detail": self.detail,
        }
