"""
Stage 6 — 학습 설정 자동 생성 (Train Config Builder).

PRD §4.6: HW 프로빙 결과와 카테고리 프리셋을 조합하여
최적의 TrainConfig를 자동 결정한다.

사용자가 "추천값 사용"을 선택하면 이 빌더의 결과를 그대로 사용.
"고급 사용자" 모드에서는 반환된 TrainConfig를 오버라이드 가능.
"""

from __future__ import annotations

import os
from pathlib import Path

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import (
    JobSpec,
    JobState,
    LoRAConfig,
    Quantization,
    RecommendedPlan,
    TaskType,
    TrainConfig,
    TrainStrategy,
)

log = get_logger(__name__)

# 태스크 타입별 LoRA target_modules 매핑
# 모델 아키텍처에 따라 다르지만 LLaMA/Mistral 계열 기준 기본값
_TASK_TARGET_MODULES: dict[TaskType, list[str]] = {
    TaskType.CHAT:           ["q_proj", "v_proj"],
    TaskType.QA:             ["q_proj", "v_proj"],
    TaskType.SUMMARY:        ["q_proj", "k_proj", "v_proj"],
    TaskType.CLASSIFICATION: ["q_proj", "v_proj"],
    TaskType.CODE:           ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# 전략별 기본 학습률
_STRATEGY_LR: dict[TrainStrategy, float] = {
    TrainStrategy.FULL_FT: 1e-5,
    TrainStrategy.LORA:    2e-4,
    TrainStrategy.QLORA:   2e-4,
    TrainStrategy.DPO:     5e-5,
    TrainStrategy.ORPO:    5e-5,
}

# 기본 체크포인트 루트
_DEFAULT_CKPT_ROOT = Path.home() / ".wfp" / "checkpoints"


class TrainConfigBuilder:
    """
    Stage 6: HW 플랜 + 카테고리 프리셋 → TrainConfig 자동 생성.

    책임:
    - RecommendedPlan 기반 전략·양자화·시퀀스 길이 결정
    - 태스크 타입 기반 LoRA 설정 결정
    - 데이터셋 크기 기반 에폭·스텝 자동 조정
    - 예상 학습 시간·메모리·디스크 사용량 추정
    - JobSpec.train_config 갱신
    """

    def build(
        self,
        spec: JobSpec,
        override: dict | None = None,
    ) -> TrainConfig:
        """
        TrainConfig 생성.

        Parameters
        ----------
        spec:
            Stage 5까지 채워진 JobSpec.
        override:
            사용자가 고급 모드에서 수동 지정한 하이퍼파라미터.
            TrainConfig 필드명을 key로 사용.

        Returns
        -------
        TrainConfig
            완성된 학습 설정.

        Raises
        ------
        WFPError(WFP-0601)
            빌드 중 치명적 오류 발생 시.
        WFPError(WFP-0602)
            지원하지 않는 전략 지정 시.
        """
        plan: RecommendedPlan | None = spec.recommended_plan

        if not plan or not plan.recommended_strategy:
            raise WFPError(
                code=WFPErrorCode.CONFIG_BUILD_FAILED,
                message="RecommendedPlan이 없습니다. Stage 2를 먼저 실행하세요.",
                stage=6,
            )

        strategy = plan.recommended_strategy
        quant = plan.recommended_quant

        log.info(
            "Stage 6: 학습 설정 빌드 시작",
            extra={
                "stage": 6,
                "job_id": str(spec.job_id),
                "extra": {"strategy": strategy.value, "quant": quant.value},
            },
        )

        # 출력 디렉토리
        ckpt_dir = str(_DEFAULT_CKPT_ROOT / str(spec.job_id))
        os.makedirs(ckpt_dir, exist_ok=True)

        # 기본 설정 결정
        config = TrainConfig(
            strategy=strategy,
            quantization=quant,
            max_seq_length=plan.max_seq_length,
            batch_size=plan.batch_size,
            gradient_accumulation=plan.gradient_accumulation,
            learning_rate=_STRATEGY_LR.get(strategy, 2e-4),
            output_dir=ckpt_dir,
        )

        # 정밀도 설정 (양자화에 따라)
        config.bf16 = quant in (Quantization.NONE,) and plan.backend != "cpu"
        config.fp16 = not config.bf16 and plan.backend == "cuda"

        # LoRA 설정 (LoRA/QLoRA 전략에서만)
        if strategy in (TrainStrategy.LORA, TrainStrategy.QLORA):
            config.lora = self._build_lora(spec, plan)

        # 데이터셋 크기 기반 에폭 조정
        config.num_epochs = self._recommend_epochs(spec.num_train)

        # 스텝 설정 (데이터셋 크기 기반)
        steps_per_epoch = max(
            1,
            spec.num_train // (config.batch_size * config.gradient_accumulation),
        )
        config.save_steps = max(50, steps_per_epoch // 4)
        config.eval_steps = config.save_steps
        config.logging_steps = max(10, steps_per_epoch // 20)

        # 사용자 오버라이드 적용
        if override:
            config = self._apply_override(config, override)

        # JobSpec 갱신
        spec.train_config = config
        spec.update_stage(stage=6, state=JobState.RUNNING)

        log.info(
            "Stage 6: 학습 설정 완료",
            extra={
                "stage": 6,
                "job_id": str(spec.job_id),
                "extra": {
                    "strategy": config.strategy.value,
                    "quant": config.quantization.value,
                    "seq_len": config.max_seq_length,
                    "batch": config.batch_size,
                    "grad_acc": config.gradient_accumulation,
                    "lr": config.learning_rate,
                    "epochs": config.num_epochs,
                    "lora_r": config.lora.r if config.lora else None,
                },
            },
        )

        return config

    def estimate_resources(self, spec: JobSpec) -> dict:
        """
        학습 리소스 추정값 반환 (UI 표시용).

        Returns
        -------
        dict
            estimated_hours, estimated_vram_gb, estimated_disk_gb
        """
        config = spec.train_config
        plan = spec.recommended_plan
        if not config or not plan:
            return {}

        # VRAM 추정: 모델 크기 × 양자화 배수 × LoRA 오버헤드
        from wfp_worker.stage4_model.model_catalog import _ALL_MODELS
        matched = next((m for m in _ALL_MODELS if m.hf_repo == spec.model_hf_repo), None)
        params_b = matched.params_b if matched else 7.0

        quant_multiplier = {
            Quantization.INT4: 0.55,
            Quantization.INT8: 1.1,
            Quantization.NONE: 2.0,
        }.get(config.quantization, 0.55)

        estimated_vram = params_b * quant_multiplier * 1.3  # 학습 오버헤드 30%

        # 디스크 추정: 체크포인트 × 에폭
        estimated_disk = params_b * 2 * config.num_epochs  # 에폭당 어댑터 저장

        # 시간 추정: 단순 선형 (토큰/초 기반 정밀 추정은 벤치마크 DB 필요)
        tokens_total = spec.num_train * config.max_seq_length * config.num_epochs
        tokens_per_sec = max(100, int(10_000 / max(params_b, 1)))  # 경험 추정
        estimated_hours = tokens_total / tokens_per_sec / 3600

        return {
            "estimated_hours": round(estimated_hours, 1),
            "estimated_vram_gb": round(estimated_vram, 1),
            "estimated_disk_gb": round(estimated_disk, 1),
        }

    # ─────────────────────────────────────────
    # 내부 구현
    # ─────────────────────────────────────────

    @staticmethod
    def _build_lora(spec: JobSpec, plan: RecommendedPlan) -> LoRAConfig:
        """태스크와 모델 크기에 맞는 LoRA 설정 생성."""
        target_modules = _TASK_TARGET_MODULES.get(spec.task_type, ["q_proj", "v_proj"])

        # VRAM이 넉넉하면 r 값 높여 표현력 증가
        from wfp_worker.stage4_model.model_catalog import _ALL_MODELS
        matched = next((m for m in _ALL_MODELS if m.hf_repo == spec.model_hf_repo), None)
        params_b = matched.params_b if matched else 7.0

        # 모델이 클수록, VRAM이 넉넉할수록 r 증가
        r = 16
        if params_b >= 13:
            r = 32
        if plan.max_vram_gb >= 24 if hasattr(plan, "max_vram_gb") else False:
            r = min(r * 2, 64)

        return LoRAConfig(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
        )

    @staticmethod
    def _recommend_epochs(num_train: int) -> int:
        """데이터 크기 기반 에폭 수 추천."""
        if num_train == 0:
            return 3
        if num_train < 500:
            return 5     # 소량 데이터 → 더 많은 에폭
        if num_train < 5000:
            return 3
        return 2          # 대량 데이터 → 과적합 방지

    @staticmethod
    def _apply_override(config: TrainConfig, override: dict) -> TrainConfig:
        """사용자 오버라이드 적용. 알 수 없는 필드는 무시."""
        data = config.model_dump()
        for key, value in override.items():
            if key in data:
                data[key] = value
            else:
                log.warning("알 수 없는 오버라이드 필드 무시", extra={"extra": {"key": key}})
        return TrainConfig(**data)
