"""
구조화 JSON 로거.

표준 필드: ts, level, service, job_id, user_id, msg, extra
Python logging.Logger를 JSON 핸들러로 래핑하여 제공.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """로그 레코드를 JSON 한 줄로 직렬화."""

    # 서비스 이름 (환경 변수나 init 시점에 설정)
    service_name: str = "wfp-worker"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "msg": record.getMessage(),
        }

        # 구조화 필드 추가 (extra= 로 전달된 값)
        for field in ("job_id", "user_id", "stage", "extra"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, service: str = "wfp-worker") -> logging.Logger:
    """
    서비스 이름이 태깅된 JSON 로거 반환.

    사용 예:
        log = get_logger(__name__)
        log.info("학습 시작", extra={"job_id": str(job.id), "stage": 7})
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 중복 핸들러 방지

    handler = logging.StreamHandler(sys.stdout)
    formatter = _JSONFormatter()
    formatter.service_name = service
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger
