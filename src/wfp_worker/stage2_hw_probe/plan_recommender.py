"""
Stage 2 — HW 능력 → 파이프라인 플랜 변환기.

ops/rules/hw_to_plan.yaml의 규칙을 순서대로 평가해
첫 번째로 매칭된 규칙의 플랜을 반환한다.

규칙 파일이 없으면 내장 기본값을 사용하여 서비스 중단 없이 동작.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import HWCaps, Quantization, RecommendedPlan, TrainStrategy

log = get_logger(__name__)

# 기본 규칙 파일 경로 (프로젝트 루트 기준)
_DEFAULT_RULES_PATH = (
    Path(__file__).parents[6] / "ops" / "rules" / "hw_to_plan.yaml"
)


class PlanRecommender:
    """
    Stage 2: HWCaps → RecommendedPlan 변환기.

    hw_to_plan.yaml 규칙을 로드하고 HWCaps에 매칭되는
    첫 번째 규칙을 선택하여 RecommendedPlan으로 반환.
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        path = rules_path or _DEFAULT_RULES_PATH
        self._rules: list[dict[str, Any]] = self._load_rules(path)

    # ─────────────────────────────────────────
    # 퍼블릭 API
    # ─────────────────────────────────────────

    def recommend(self, caps: HWCaps) -> RecommendedPlan:
        """
        HWCaps를 받아 첫 번째 매칭 규칙의 플랜을 반환.

        Raises
        ------
        WFPError(WFP-0203)
            어떤 규칙도 매칭되지 않을 때 (규칙 파일 오류 등).
        """
        for rule in self._rules:
            if self._matches(rule["condition"], caps):
                plan = self._build_plan(rule)
                log.info(
                    "HW 플랜 결정",
                    extra={
                        "stage": 2,
                        "extra": {
                            "rule_id": rule["id"],
                            "max_model_params_b": plan.max_model_params_b,
                            "strategy": plan.recommended_strategy,
                            "backend": plan.backend,
                            "warning": plan.warning,
                        },
                    },
                )
                return plan

        raise WFPError(
            code=WFPErrorCode.NO_PLAN_MATCH,
            message="HW 사양에 매칭되는 플랜 규칙이 없습니다.",
            stage=2,
            detail=f"VRAM={caps.max_vram_gb}GB, RAM={caps.ram_gb}GB, OS={caps.os}",
        )

    # ─────────────────────────────────────────
    # 내부 구현
    # ─────────────────────────────────────────

    @staticmethod
    def _load_rules(path: Path) -> list[dict[str, Any]]:
        """YAML 규칙 파일 로드. 파일 없으면 최소 내장 규칙 반환."""
        if path.exists():
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            rules = data.get("rules", [])
            log.debug("HW 규칙 로드", extra={"extra": {"path": str(path), "count": len(rules)}})
            return rules

        log.warning(
            "hw_to_plan.yaml 파일 없음 — 내장 기본 규칙 사용",
            extra={"extra": {"tried_path": str(path)}},
        )
        return _BUILTIN_FALLBACK_RULES

    @staticmethod
    def _matches(condition: dict[str, Any], caps: HWCaps) -> bool:
        """
        단일 규칙 조건을 HWCaps에 적용.

        조건 키:
          gpu_vendor        : "nvidia" | "amd" | "apple" | "none"
          vram_gb_min/max   : NVIDIA VRAM 범위 (GB)
          unified_memory_gb_min/max : Apple Unified Memory 범위
          ram_gb_min        : CPU-only RAM 최소값
        """
        # GPU 벤더 확인
        if "gpu_vendor" in condition:
            vendor = condition["gpu_vendor"]
            if vendor == "none":
                # GPU 없는 경우: gpus 리스트 비어 있거나 vendor=="none"
                has_real_gpu = any(g.vendor not in ("none", "unknown", "") for g in caps.gpus)
                if has_real_gpu:
                    return False
            else:
                # 특정 벤더 GPU가 있어야 함
                matching = [g for g in caps.gpus if g.vendor == vendor]
                if not matching:
                    return False

                # VRAM 범위 조건 (NVIDIA 전용)
                vram = max(g.vram_gb for g in matching)
                if "vram_gb_min" in condition and vram < condition["vram_gb_min"]:
                    return False
                if "vram_gb_max" in condition and vram > condition["vram_gb_max"]:
                    return False

                # Apple Unified Memory 범위
                if vendor == "apple":
                    umem = caps.unified_memory_gb
                    if "unified_memory_gb_min" in condition and umem < condition["unified_memory_gb_min"]:
                        return False
                    if "unified_memory_gb_max" in condition and umem > condition["unified_memory_gb_max"]:
                        return False

        # RAM 최소 조건 (CPU 전용 규칙에서 사용)
        if "ram_gb_min" in condition and caps.ram_gb < condition["ram_gb_min"]:
            return False

        return True

    @staticmethod
    def _build_plan(rule: dict[str, Any]) -> RecommendedPlan:
        """규칙 dict를 RecommendedPlan 모델로 변환."""
        p = rule["plan"]

        # 전략 목록
        raw_strategies = p.get("strategies", [])
        strategies = []
        for s in raw_strategies:
            try:
                strategies.append(TrainStrategy(s))
            except ValueError:
                pass

        # 양자화 목록
        raw_quants = p.get("quantization", [])
        quants = []
        for q in raw_quants:
            try:
                quants.append(Quantization(q))
            except ValueError:
                pass

        # 권장 전략
        rec_strategy_raw = p.get("recommended_strategy")
        rec_strategy: TrainStrategy | None = None
        if rec_strategy_raw:
            try:
                rec_strategy = TrainStrategy(rec_strategy_raw)
            except ValueError:
                pass

        # 권장 양자화
        rec_quant_raw = p.get("recommended_quant", "int4")
        try:
            rec_quant = Quantization(rec_quant_raw)
        except ValueError:
            rec_quant = Quantization.INT4

        # bf16인 경우 Quantization.NONE으로 처리
        if rec_quant_raw == "bf16":
            rec_quant = Quantization.NONE

        return RecommendedPlan(
            rule_id=rule.get("id", ""),
            max_model_params_b=float(p.get("max_model_params_b", 0)),
            strategies=strategies,
            quantization=quants,
            recommended_strategy=rec_strategy,
            recommended_quant=rec_quant,
            backend=p.get("backend", "cuda"),
            max_seq_length=int(p.get("max_seq_length", 2048)),
            batch_size=int(p.get("batch_size", 1)),
            gradient_accumulation=int(p.get("gradient_accumulation", 8)),
            warning=p.get("warning", ""),
        )


# ─────────────────────────────────────────
# 규칙 파일 없을 때 내장 폴백 (최소 집합)
# ─────────────────────────────────────────
_BUILTIN_FALLBACK_RULES: list[dict[str, Any]] = [
    {
        "id": "builtin_nvidia_24gb",
        "condition": {"gpu_vendor": "nvidia", "vram_gb_min": 24},
        "plan": {
            "max_model_params_b": 13, "strategies": ["qlora"],
            "quantization": ["int4"], "recommended_strategy": "qlora",
            "recommended_quant": "int4", "max_seq_length": 2048,
            "batch_size": 2, "gradient_accumulation": 8,
        },
    },
    {
        "id": "builtin_nvidia_any",
        "condition": {"gpu_vendor": "nvidia", "vram_gb_min": 1},
        "plan": {
            "max_model_params_b": 7, "strategies": ["qlora"],
            "quantization": ["int4"], "recommended_strategy": "qlora",
            "recommended_quant": "int4", "max_seq_length": 1024,
            "batch_size": 1, "gradient_accumulation": 16,
        },
    },
    {
        "id": "builtin_apple",
        "condition": {"gpu_vendor": "apple"},
        "plan": {
            "max_model_params_b": 7, "strategies": ["qlora"],
            "quantization": ["int4"], "recommended_strategy": "qlora",
            "recommended_quant": "int4", "backend": "mps",
            "max_seq_length": 2048, "batch_size": 1, "gradient_accumulation": 8,
        },
    },
    {
        "id": "builtin_cpu_fallback",
        "condition": {"gpu_vendor": "none"},
        "plan": {
            "max_model_params_b": 1, "strategies": ["qlora"],
            "quantization": ["int4"], "recommended_strategy": "qlora",
            "recommended_quant": "int4", "backend": "cpu",
            "max_seq_length": 512, "batch_size": 1, "gradient_accumulation": 32,
            "warning": "CPU-only 학습은 매우 느립니다.",
        },
    },
]
