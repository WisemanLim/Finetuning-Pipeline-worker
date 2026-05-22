"""
데이터 업로드 라우터.

엔드포인트:
  POST /data/upload   사용자 학습 데이터 멀티파트 업로드 → 서버 측 경로 반환

업로드된 파일은 ~/.wfp/uploads/<user_id>/<uuid>_<original_name> 에 저장.
JobCreateRequest.data_source_path 에 이 경로를 그대로 사용.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from wfp_worker.auth import get_current_user
from wfp_worker.common.logger import get_logger
from wfp_worker.db import User

log = get_logger(__name__)
router = APIRouter(prefix="/data", tags=["data"])

# 허용 확장자 (collector._EXT_MAP과 동기)
_ALLOWED_EXTS = {".jsonl", ".json", ".csv", ".parquet", ".txt", ".md",
                 ".html", ".htm", ".pdf", ".docx"}

# 최대 파일 크기 — 1GB (운영은 .env로 외부화)
_MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024


def _upload_dir(user_id: str) -> Path:
    base = Path.home() / ".wfp" / "uploads" / user_id
    base.mkdir(parents=True, exist_ok=True)
    return base


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_data(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> dict:
    """
    학습 데이터 파일 업로드.

    반환:
        path     서버 측 절대 경로 (JobCreateRequest.data_source_path에 사용)
        filename 원본 파일명
        size_bytes 저장된 크기
    """
    if not file.filename:
        raise HTTPException(400, detail="No filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            400,
            detail=f"지원하지 않는 확장자: {ext}. 허용: {sorted(_ALLOWED_EXTS)}",
        )

    # 사용자별 임시 디렉토리
    target_dir = _upload_dir(user.id)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    target = target_dir / safe_name

    # 스트리밍 저장 + 크기 제한
    total = 0
    with target.open("wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB 청크
            total += len(chunk)
            if total > _MAX_FILE_BYTES:
                f.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    detail=f"파일 크기 초과: {_MAX_FILE_BYTES // (1024**3)}GB 제한",
                )
            f.write(chunk)

    log.info(
        "data_upload",
        extra={"user_id": user.id, "extra": {"path": str(target), "size": total, "ext": ext}},
    )

    return {
        "path": str(target),
        "filename": file.filename,
        "size_bytes": total,
    }
