"""
Stage 4 — 카테고리별 베이스 모델 카탈로그.

PRD §3.2 카테고리별 파이프라인 프리셋을 코드로 구현.
운영 환경에서는 DB(category_preset)에서 로드하도록 교체 가능.

각 항목:
  hf_repo   : Hugging Face repo id
  params_b  : 파라미터 수 (단위: B)
  language  : 지원 언어 목록
  license   : 라이선스 식별자
  hw_min_vram_gb : 권장 최소 VRAM (GB)
  tags      : 분류 태그 (code, medical, finance 등)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelEntry:
    """단일 모델 카탈로그 항목."""

    hf_repo: str
    params_b: float
    language: list[str]
    license: str
    hw_min_vram_gb: float  # 학습 가능 최소 VRAM (QLoRA 4bit 기준)
    tags: list[str] = field(default_factory=list)
    priority: int = 5       # 낮을수록 우선순위 높음 (1 = 최우선)

    def __hash__(self) -> int:
        return hash(self.hf_repo)


# ─────────────────────────────────────────
# 전체 모델 풀 (PRD §3.2 + 검증된 오픈소스)
# ─────────────────────────────────────────

_ALL_MODELS: list[ModelEntry] = [
    # ── 범용 한국어 강화 ──
    ModelEntry("Qwen/Qwen2-7B-Instruct",           7.0,  ["ko","en","zh"], "apache-2.0", 6.0,  ["general","ko"], 1),
    ModelEntry("Qwen/Qwen2-14B-Instruct",          14.0, ["ko","en","zh"], "apache-2.0", 12.0, ["general","ko"], 2),
    ModelEntry("meta-llama/Llama-3-8B-Instruct",    8.0,  ["ko","en"],      "llama3",     6.0,  ["general"],       2),
    ModelEntry("mistralai/Mistral-7B-Instruct-v0.3", 7.0, ["en"],           "apache-2.0", 6.0,  ["general"],       3),
    ModelEntry("google/gemma-2-9b-it",               9.0,  ["en"],           "gemma",      8.0,  ["general"],       3),
    ModelEntry("microsoft/Phi-3-medium-4k-instruct", 14.0, ["en"],           "mit",        12.0, ["general"],       4),

    # ── 코드 특화 (J 정보통신) ──
    ModelEntry("codellama/CodeLlama-13b-Instruct-hf", 13.0, ["en"], "llama2", 10.0, ["code"], 1),
    ModelEntry("deepseek-ai/deepseek-coder-6.7b-instruct", 6.7, ["en","zh"], "deepseek", 6.0, ["code"], 2),
    ModelEntry("Qwen/Qwen2.5-Coder-7B-Instruct",    7.0,  ["ko","en","zh"], "apache-2.0", 6.0,  ["code","ko"], 1),

    # ── 의료 특화 (Q 보건) ──
    ModelEntry("BioMistral/BioMistral-7B",           7.0,  ["en"],           "llama2",     6.0,  ["medical"], 1),
    ModelEntry("epfl-llm/meditron-7b",               7.0,  ["en"],           "llama2",     6.0,  ["medical"], 2),

    # ── 소형 / Apple Silicon 호환 ──
    ModelEntry("Qwen/Qwen2-1.5B-Instruct",           1.5,  ["ko","en","zh"], "apache-2.0", 2.0,  ["small","ko"], 1),
    ModelEntry("microsoft/Phi-3-mini-4k-instruct",   3.8,  ["en"],           "mit",        3.0,  ["small"],       2),
    ModelEntry("google/gemma-2-2b-it",               2.0,  ["en"],           "gemma",      2.0,  ["small"],       3),
]


class ModelCatalog:
    """
    카테고리(KSIC section)와 HW 제약으로 후보 모델 목록 반환.

    필터링 기준:
      1. KSIC 섹션에 맞는 태그 우선
      2. VRAM 제약 (max_vram_gb 이상 hw_min_vram_gb인 모델 제외)
      3. 언어 필터 (language 목록에 요청 언어 포함 여부)
      4. priority 오름차순 정렬
    """

    # 섹션 → 선호 태그 매핑 (PRD §3.2)
    _SECTION_TAGS: dict[str, list[str]] = {
        "J": ["code"],
        "Q": ["medical"],
        "K": ["finance"],
        "C": ["general"],
        "G": ["general"],
        "I": ["general"],
        "M": ["general"],
        "P": ["general"],
        "F": ["general"],
    }

    def get_candidates(
        self,
        ksic_section: str,
        max_vram_gb: float,
        language: str = "ko",
        max_results: int = 5,
    ) -> list[ModelEntry]:
        """
        필터링·정렬된 후보 모델 목록 반환.

        Parameters
        ----------
        ksic_section:
            KSIC 대분류 (A~U).
        max_vram_gb:
            사용 가능한 최대 VRAM(GB). 0이면 CPU 환경.
        language:
            목표 언어 코드 (예: "ko").
        max_results:
            반환 최대 수.
        """
        preferred_tags = self._SECTION_TAGS.get(ksic_section, ["general"])

        # VRAM 필터: CPU(0GB)이면 hw_min_vram_gb <= 4 모델만
        effective_vram = max_vram_gb if max_vram_gb > 0 else 4.0

        candidates: list[tuple[int, int, ModelEntry]] = []
        for model in _ALL_MODELS:
            if model.hw_min_vram_gb > effective_vram:
                continue

            # 태그 점수 (섹션과 관련 태그 있으면 -10 보너스)
            tag_bonus = -10 if any(t in model.tags for t in preferred_tags) else 0

            # 언어 점수 (요청 언어 지원 시 -5 보너스)
            lang_bonus = -5 if language in model.language else 0

            sort_key = (model.priority + tag_bonus + lang_bonus, model.params_b)
            candidates.append((sort_key[0], int(sort_key[1] * 10), model))

        candidates.sort(key=lambda x: (x[0], x[1]))
        return [c[2] for c in candidates[:max_results]]
