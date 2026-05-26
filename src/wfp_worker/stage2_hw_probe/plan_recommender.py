"""
Stage 2 — HW 능력 → 파이프라인 플랜 변환기.

ops/rules/hw_to_plan.yaml의 규칙을 순서대로 평가해
첫 번째로 매칭된 규칙의 플랜을 반환한다.

규칙 파일이 없으면 내장 기본값을 사용하여 서비스 중단 없이 동작.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import yaml

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import HWCaps, Quantization, RecommendedPlan, TrainStrategy

log = get_logger(__name__)


def _discover_rules_path() -> Path | None:
    """hw_to_plan.yaml 경로 자동 탐색 — 다양한 배포 형태 지원.

    우선순위:
      1. WFP_RULES_PATH 환경 변수
      2. Docker 컨테이너 마운트 경로 (/app/ops/rules/...)
      3. Sibling repo 구조 (worker와 ops가 형제 디렉토리)
      4. 모노레포 구조 (worker repo 내부에 ops/)
      5. 내장 기본 규칙 (코드)
    """
    env_path = os.environ.get("WFP_RULES_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # __file__ = .../<worker>/.venv/lib/python3.12/site-packages/wfp_worker/stage2_hw_probe/plan_recommender.py
    # parents[6] = worker repo root
    worker_root = Path(__file__).parents[6]

    candidates = [
        Path("/app/ops/rules/hw_to_plan.yaml"),                            # Docker mount
        worker_root.parent / "Finetuning-Pipeline-ops" / "rules" / "hw_to_plan.yaml",  # sibling repo
        worker_root / "ops" / "rules" / "hw_to_plan.yaml",                 # monorepo (legacy)
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


_DEFAULT_RULES_PATH = _discover_rules_path()


class PlanRecommender:
    """
    Stage 2: HWCaps → RecommendedPlan 변환기.

    hw_to_plan.yaml 규칙을 로드하고 HWCaps에 매칭되는
    첫 번째 규칙을 선택하여 RecommendedPlan으로 반환.
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        # 매 호출마다 재탐색 — env var/파일이 런타임에 바뀔 수 있음.
        # 발견 실패 시 _load_rules 내부에서 _BUILTIN_FALLBACK_RULES 사용.
        path = rules_path or _discover_rules_path() or Path("/nonexistent/hw_to_plan.yaml")
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

        # 상세 정보 — frontend 강제 진행 확인 모달용 + WS event용
        gpu_info = (
            f"{caps.gpus[0].vendor}/{caps.gpus[0].name}/{caps.gpus[0].vram_gb}GB"
            if caps.gpus else "none"
        )
        tried_ids = [r.get("id", "?") for r in self._rules]
        raise WFPError(
            code=WFPErrorCode.NO_PLAN_MATCH,
            message="HW 사양에 매칭되는 플랜 규칙이 없습니다.",
            stage=2,
            detail=(
                f"OS={caps.os}/{caps.arch}, RAM={caps.ram_gb}GB, "
                f"GPU={gpu_info}, max_VRAM={caps.max_vram_gb}GB, "
                f"CUDA={caps.cuda_available}, MPS={caps.mps_available}. "
                f"검토한 규칙 {len(self._rules)}개: {', '.join(tried_ids[:10])}"
                f"{'...' if len(tried_ids) > 10 else ''}"
            ),
        )

    def force_fallback_plan(self, caps: HWCaps) -> RecommendedPlan:
        """매칭 실패 시 사용자 동의로 강제 진행 — HW 기반 최소 안전 plan 생성.

        규칙 매칭 실패 + 사용자 confirm → 이 메서드로 fallback plan 즉석 생성.
        WS warning event로 사용자에게 표시 권장.
        """
        has_gpu = bool(caps.gpus)
        cuda = caps.cuda_available
        mps = caps.mps_available

        if has_gpu and cuda:
            backend = "cuda"
            # vram 0 (unified memory)이면 ram 기반, 그 외 vram 기반
            mem_gb = caps.max_vram_gb if caps.max_vram_gb >= 4 else caps.ram_gb
            if mem_gb >= 60:
                params, seq, batch = 13.0, 2048, 2
            elif mem_gb >= 24:
                params, seq, batch = 8.0, 2048, 2
            else:
                params, seq, batch = 3.0, 1024, 1
        elif has_gpu and mps:
            backend, params, seq, batch = "mps", 3.0, 1024, 1
        else:
            backend, params, seq, batch = "cpu", 1.0, 512, 1

        log.warning(
            "강제 fallback plan 생성 — 사용자 동의로 매칭 규칙 우회",
            extra={"stage": 2, "extra": {
                "backend": backend, "max_params_b": params,
                "ram_gb": caps.ram_gb, "vram_gb": caps.max_vram_gb,
            }},
        )

        return self._build_plan({
            "id": "force_fallback_user_consent",
            "plan": {
                "max_model_params_b": params,
                "strategies": ["qlora"],
                "quantization": ["int4"],
                "recommended_strategy": "qlora",
                "recommended_quant": "int4",
                "backend": backend,
                "max_seq_length": seq,
                "batch_size": batch,
                "gradient_accumulation": 8 if backend != "cpu" else 32,
                "warning": "사용자 동의로 강제 진행 — 표준 규칙 매칭 실패. 학습 결과 보장 안 됨.",
            },
        })

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
    # NVIDIA Unified Memory (Grace/Blackwell, Jetson, DGX Spark) — vram=0 보고됨
    {
        "id": "builtin_nvidia_unified_memory",
        "condition": {"gpu_vendor": "nvidia", "vram_gb_max": 1, "ram_gb_min": 32},
        "plan": {
            "max_model_params_b": 34, "strategies": ["lora", "qlora"],
            "quantization": ["bf16", "int4"], "recommended_strategy": "qlora",
            "recommended_quant": "int4", "backend": "cuda",
            "max_seq_length": 4096, "batch_size": 2, "gradient_accumulation": 8,
        },
    },
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
