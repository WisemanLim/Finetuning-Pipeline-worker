"""
모델 카탈로그 조회 라우터.

엔드포인트:
  GET /models?section=J&max_vram_gb=16&language=ko
    → ml-frontend Step 2 위자드 후보 목록 제공.

소스 정책:
  1. DB의 model_catalog가 존재하면 그것을 1차 사용
  2. 없으면 stage4_model.model_catalog._ALL_MODELS의 시드 데이터 폴백
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from wfp_worker.db import ModelCatalogEntry, get_db
from wfp_worker.stage4_model.model_catalog import ModelCatalog, _ALL_MODELS

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    hf_repo: str
    params_b: float
    language: list[str]
    license: str
    hw_min_vram_gb: float
    tags: list[str]


@router.get("/", response_model=list[ModelInfo])
def list_models(
    section: str = Query(default="", description="KSIC 대분류 (A~U)"),
    max_vram_gb: float = Query(default=0.0, ge=0.0),
    language: str = Query(default="ko"),
    max_results: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> list[ModelInfo]:
    """
    HW 조건 + 카테고리 기반 추천 모델 목록.

    Stage 2 (HW Probe) 결과의 max_vram_gb를 전달하면 학습 가능한 모델만 반환.
    """
    # DB에 활성 카탈로그가 있으면 우선 사용
    db_entries = db.query(ModelCatalogEntry).filter(ModelCatalogEntry.is_active.is_(True)).all()

    if db_entries:
        from dataclasses import replace
        from wfp_worker.stage4_model.model_catalog import ModelEntry

        catalog = ModelCatalog()
        # 데이터베이스 항목을 ModelEntry 형태로 변환하여 필터
        custom_entries: list[ModelEntry] = []
        for e in db_entries:
            custom_entries.append(ModelEntry(
                hf_repo=e.hf_repo,
                params_b=e.params_b,
                language=[s.strip() for s in e.language.split(",") if s.strip()],
                license=e.license,
                hw_min_vram_gb=e.hw_min_vram_gb,
                tags=[s.strip() for s in e.tags.split(",") if s.strip()],
                priority=e.priority,
            ))
        candidates = _filter_models(custom_entries, section, max_vram_gb, language, max_results)
    else:
        # 시드 데이터 폴백
        candidates = ModelCatalog().get_candidates(
            ksic_section=section,
            max_vram_gb=max_vram_gb,
            language=language,
            max_results=max_results,
        )

    return [
        ModelInfo(
            hf_repo=m.hf_repo,
            params_b=m.params_b,
            language=m.language,
            license=m.license,
            hw_min_vram_gb=m.hw_min_vram_gb,
            tags=m.tags,
        )
        for m in candidates
    ]


def _filter_models(entries, section, max_vram_gb, language, max_results):
    """DB 항목용 간단 필터 — ModelCatalog 로직과 동일."""
    effective_vram = max_vram_gb if max_vram_gb > 0 else 4.0
    section_tags = ModelCatalog._SECTION_TAGS.get(section, ["general"])

    candidates = []
    for m in entries:
        if m.hw_min_vram_gb > effective_vram:
            continue
        tag_bonus = -10 if any(t in m.tags for t in section_tags) else 0
        lang_bonus = -5 if language in m.language else 0
        candidates.append((m.priority + tag_bonus + lang_bonus, m.params_b, m))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [c[2] for c in candidates[:max_results]]
