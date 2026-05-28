"""
Catalog Auto-Refresher
======================

운영 중 모델 카탈로그/카테고리 프리셋을 자동 추가·갱신.

흐름
----
1. HF Hub trending 검색 (카테고리별 키워드) → 신규 모델 후보 수집
2. 라이선스 safelist + params 범위 + 다운로드 임계로 필터
3. ModelCatalogEntry 에 없으면 INSERT (기존 행은 사용자가 편집했을 수 있어 손대지 않음)
4. CategoryPreset 의 추천 모델을 현재 카탈로그 + KSIC 섹션 태그 매핑으로 재계산
   - 기존 프리셋은 recommended_models 만 갱신 (이름·설명은 운영자 권한 보존)
   - 시드된 KSIC 섹션 중 프리셋이 없으면 새로 생성

LLM/외부 API 키 불필요. huggingface_hub 만 의존.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from wfp_worker.db import CategoryPreset, ModelCatalogEntry

log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

# 라이선스 safelist — 운영 사용 가능한 오픈 라이선스만 신규 추가.
_LICENSE_SAFELIST: set[str] = {
    "apache-2.0", "mit", "llama2", "llama3", "llama3.1", "llama3.2",
    "gemma", "deepseek", "bsd-3-clause",
}

# HF Hub 검색 결과 → 다운로드 임계. 잡음 모델 제거.
_MIN_DOWNLOADS = 500

# 모델 크기 허용 범위 (params B).
_MIN_PARAMS_B = 0.5
_MAX_PARAMS_B = 70.0

# 카테고리별 신규 모델 N건 추가 한도 (한 번 호출당).
_MAX_NEW_PER_CATEGORY = 5

# 프리셋 1개당 추천 모델 캡.
_PRESET_MAX_MODELS = 5

# KSIC 섹션별 검색 키워드 + 자동 태그 매핑.
@dataclass(frozen=True)
class SectionSpec:
    section: str
    name: str         # 기본 프리셋 이름 (없으면 생성)
    task_type: str
    description: str
    tags: tuple[str, ...]      # 카탈로그 등록 시 추가할 태그
    queries: tuple[str, ...]   # HF Hub 검색 키워드


_SECTION_SPECS: list[SectionSpec] = [
    SectionSpec("C", "제조업 기본", "chat",
                "제조업 — 매뉴얼/규격 기반 챗",
                ("general",),
                ("instruct 7b", "instruct chat")),
    SectionSpec("F", "건설", "chat",
                "건설표준시방서/KS",
                ("general",),
                ("instruct chat",)),
    SectionSpec("G", "도매·소매 기본", "chat",
                "상품 카탈로그/FAQ instruction",
                ("general",),
                ("instruct chat",)),
    SectionSpec("I", "숙박·음식점 기본", "chat",
                "메뉴/리뷰 톤·매너 튜닝",
                ("general",),
                ("instruct chat",)),
    SectionSpec("J", "정보통신·코드", "code",
                "코드 특화",
                ("code",),
                ("coder instruct", "code llama", "deepseek coder")),
    SectionSpec("K", "금융·보험", "chat",
                "공시/약관 — 안전성 평가 강화",
                ("general", "finance"),
                ("finance instruct", "finance llm")),
    SectionSpec("M", "전문·과학·기술", "chat",
                "논문/특허 긴 컨텍스트",
                ("general",),
                ("instruct long context",)),
    SectionSpec("P", "교육", "chat",
                "교과서/문제은행 난이도 조절",
                ("general",),
                ("instruct chat",)),
    SectionSpec("Q", "보건·의료", "qa",
                "임상가이드 — 근거성 평가 강화",
                ("medical",),
                ("medical instruct", "biomistral", "meditron")),
]


# ─────────────────────────────────────────
# 결과 타입
# ─────────────────────────────────────────


@dataclass
class RefreshSummary:
    models_added: int = 0
    models_skipped: int = 0
    presets_added: int = 0
    presets_updated: int = 0
    presets_unchanged: int = 0
    hf_search_errors: int = 0
    queries_run: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "models_added": self.models_added,
            "models_skipped": self.models_skipped,
            "presets_added": self.presets_added,
            "presets_updated": self.presets_updated,
            "presets_unchanged": self.presets_unchanged,
            "hf_search_errors": self.hf_search_errors,
            "queries_run": self.queries_run,
        }


# ─────────────────────────────────────────
# HF Hub 유틸
# ─────────────────────────────────────────


_PARAMS_TAG_RE = re.compile(r"^(\d+(?:\.\d+)?)b$", re.IGNORECASE)
_PARAMS_REPO_RE = re.compile(r"(\d+(?:\.\d+)?)b(?:[-_/.]|$)", re.IGNORECASE)


def _parse_params_b(model_info: object) -> float | None:
    """HF ModelInfo → 추정 파라미터(B)."""
    tags = getattr(model_info, "tags", None) or []
    for t in tags:
        m = _PARAMS_TAG_RE.match(str(t).strip())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    repo = getattr(model_info, "id", None) or getattr(model_info, "modelId", None) or ""
    m = _PARAMS_REPO_RE.search(str(repo).lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _infer_license(model_info: object) -> str | None:
    """HF ModelInfo → 라이선스 소문자. card_data → tags 순."""
    cd = getattr(model_info, "card_data", None) or getattr(model_info, "cardData", None)
    if cd is not None:
        lic = getattr(cd, "license", None) if not isinstance(cd, dict) else cd.get("license")
        if lic:
            return str(lic).strip().lower()
    tags = getattr(model_info, "tags", None) or []
    for t in tags:
        s = str(t).strip().lower()
        if s.startswith("license:"):
            return s.split(":", 1)[1]
    return None


def _infer_tags(model_info: object, repo: str) -> list[str]:
    """모델의 본질 태그 추정 — HF tags + repo 이름. code/medical/finance/general."""
    hf_tags = {str(t).lower() for t in (getattr(model_info, "tags", None) or [])}
    name = repo.lower()
    tags: list[str] = []
    if "code" in hf_tags or any(k in name for k in ("coder", "codellama", "starcoder")):
        tags.append("code")
    if (
        "medical" in hf_tags
        or "clinical" in hf_tags
        or any(k in name for k in ("biomistral", "meditron", "medalpaca", "clinical"))
    ):
        tags.append("medical")
    if "finance" in hf_tags or "finance" in name or "fingpt" in name:
        tags.append("finance")
    if not tags:
        tags.append("general")
    return tags


def _infer_languages(model_info: object, repo: str) -> list[str]:
    """언어 추정 — 태그 / repo 이름 기반."""
    langs: list[str] = []
    tags = [str(t).lower() for t in (getattr(model_info, "tags", None) or [])]
    for code in ("ko", "en", "zh", "ja"):
        if code in tags or f"language:{code}" in tags:
            langs.append(code)
    if not langs:
        # 한국어 친화 vendor 추정
        lname = repo.lower()
        if any(k in lname for k in ("qwen", "polyglot-ko", "kogpt", "llama-3", "llama3")):
            langs = ["ko", "en"]
        else:
            langs = ["en"]
    return langs


def _estimate_vram_gb(params_b: float) -> float:
    """QLoRA 4bit 학습 최소 VRAM 추정. 보수적으로 floor 2GB."""
    raw = params_b * 0.85 + 1.0
    return max(2.0, math.ceil(raw * 2) / 2.0)


def _hf_list(query: str, limit: int) -> list[object]:
    """HF Hub list_models 호출. 실패 시 빈 리스트."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub not installed — skip HF search")
        return []
    try:
        api = HfApi()
        return list(api.list_models(
            search=query, limit=limit, sort="downloads", direction=-1,
        ))
    except Exception as e:
        log.warning("hf_list_failed query=%r err=%s", query, e)
        return []


# ─────────────────────────────────────────
# 모델 갱신
# ─────────────────────────────────────────


def _refresh_models(db: Session, summary: RefreshSummary) -> dict[str, ModelCatalogEntry]:
    """
    HF Hub trending 으로 신규 모델 추가. 기존 행은 절대 수정하지 않음
    (운영자가 편집했을 수 있음 → 보존). 반환: {hf_repo: entry} 활성 카탈로그.
    """
    existing: dict[str, ModelCatalogEntry] = {
        e.hf_repo: e for e in db.query(ModelCatalogEntry).all()
    }

    for spec in _SECTION_SPECS:
        added_this_section = 0
        for q in spec.queries:
            summary.queries_run += 1
            results = _hf_list(q, limit=20)
            if not results:
                summary.hf_search_errors += 1
                continue
            for m in results:
                if added_this_section >= _MAX_NEW_PER_CATEGORY:
                    break
                repo = getattr(m, "id", None) or getattr(m, "modelId", None) or ""
                repo = str(repo).strip()
                if not repo or repo in existing:
                    summary.models_skipped += 1
                    continue
                downloads = int(getattr(m, "downloads", 0) or 0)
                if downloads < _MIN_DOWNLOADS:
                    summary.models_skipped += 1
                    continue
                params_b = _parse_params_b(m)
                if params_b is None or not (_MIN_PARAMS_B <= params_b <= _MAX_PARAMS_B):
                    summary.models_skipped += 1
                    continue
                lic = _infer_license(m) or "unknown"
                if lic not in _LICENSE_SAFELIST:
                    summary.models_skipped += 1
                    continue
                langs = _infer_languages(m, repo)
                tags = _infer_tags(m, repo)
                # 섹션이 명시한 도메인 태그가 모델 본질과 일치하면 합집합
                for st in spec.tags:
                    if st not in tags and st in {"code", "medical", "finance"}:
                        # 본질 태그 추정에서 누락된 도메인일 때만 보강
                        if st in {str(t).lower() for t in (getattr(m, "tags", None) or [])}:
                            tags.append(st)
                if "ko" in langs and "ko" not in tags:
                    tags.append("ko")
                entry = ModelCatalogEntry(
                    hf_repo=repo,
                    params_b=params_b,
                    license=lic,
                    hw_min_vram_gb=_estimate_vram_gb(params_b),
                    language=",".join(langs),
                    tags=",".join(tags),
                    priority=3,
                )
                db.add(entry)
                existing[repo] = entry
                summary.models_added += 1
                added_this_section += 1
            if added_this_section >= _MAX_NEW_PER_CATEGORY:
                break

    if summary.models_added:
        db.flush()
    return existing


# ─────────────────────────────────────────
# 프리셋 갱신
# ─────────────────────────────────────────


def _score_for_preset(entry: ModelCatalogEntry, spec: SectionSpec) -> tuple[int, float, int]:
    """프리셋 내 정렬 키 — (priority bonus, params 거리, repo 안정성)."""
    entry_tags = {t.strip() for t in entry.tags.split(",") if t.strip()}
    tag_bonus = -10 if any(t in entry_tags for t in spec.tags) else 0
    # 7B~9B 선호 — 거리 계산
    params_dist = abs(entry.params_b - 8.0)
    # 안정성 — 우선순위
    return (entry.priority + tag_bonus, params_dist, entry.priority)


def _refresh_presets(
    db: Session, catalog: dict[str, ModelCatalogEntry], summary: RefreshSummary
) -> None:
    """KSIC 섹션별 프리셋 recommended_models 재계산."""
    existing_presets: dict[tuple[str, str], CategoryPreset] = {
        (p.ksic_section, p.name): p for p in db.query(CategoryPreset).all()
    }
    # 이름 단독으로도 매칭 (시드와 이름은 같지만 섹션 변경 케이스)
    by_name: dict[str, CategoryPreset] = {p.name: p for p in existing_presets.values()}

    active = [e for e in catalog.values() if e.is_active]

    for spec in _SECTION_SPECS:
        # spec.tags 와 1개 이상 매칭되는 활성 모델
        spec_tag_set = set(spec.tags)
        candidates = [
            e for e in active
            if spec_tag_set & {t.strip() for t in e.tags.split(",") if t.strip()}
        ]
        if not candidates:
            # general 폴백 — 섹션이 general 이 아닌 경우에도 빈 리스트는 피함
            candidates = [
                e for e in active
                if "general" in {t.strip() for t in e.tags.split(",") if t.strip()}
            ]
        candidates.sort(key=lambda e: _score_for_preset(e, spec))
        top = candidates[:_PRESET_MAX_MODELS]
        new_csv = ",".join(e.hf_repo for e in top)

        preset = existing_presets.get((spec.section, spec.name)) or by_name.get(spec.name)
        if preset is None:
            db.add(CategoryPreset(
                name=spec.name,
                ksic_section=spec.section,
                task_type=spec.task_type,
                recommended_models=new_csv,
                default_config_json="{}",
                description=spec.description,
            ))
            summary.presets_added += 1
            continue
        if preset.recommended_models == new_csv:
            summary.presets_unchanged += 1
            continue
        preset.recommended_models = new_csv
        # task_type/description 은 운영자 편집 보존 — 비어있을 때만 채움
        if not preset.task_type:
            preset.task_type = spec.task_type
        if not preset.description:
            preset.description = spec.description
        summary.presets_updated += 1


# ─────────────────────────────────────────
# 공개 진입점
# ─────────────────────────────────────────


def refresh_catalog(db: Session) -> RefreshSummary:
    """
    카탈로그 + 프리셋 한 번에 자동 갱신.

    트랜잭션: 호출자가 commit 책임. (admin endpoint / CLI 가 commit.)
    """
    summary = RefreshSummary()
    catalog = _refresh_models(db, summary)
    _refresh_presets(db, catalog, summary)
    return summary
