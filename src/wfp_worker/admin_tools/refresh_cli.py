"""
CLI: 카탈로그/프리셋 자동 갱신.

사용:
    python -m wfp_worker.admin_tools.refresh_cli
    make catalog-refresh

환경 변수:
    WFP_DB_URL  — SQLAlchemy URL (db 모듈 기본값 사용 시 생략 가능)
"""

from __future__ import annotations

import json
import sys

from wfp_worker.admin_tools.catalog_refresher import refresh_catalog
from wfp_worker.db import SessionLocal


def main() -> int:
    db = SessionLocal()
    try:
        summary = refresh_catalog(db)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[error] refresh failed: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
