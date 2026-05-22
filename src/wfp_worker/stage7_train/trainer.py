"""
Stage 7 — 학습 실행 (Trainer).

PRD §4.7: transformers.Trainer + trl.SFTTrainer 기반 학습.

실제 학습은 torch + transformers + peft + trl이 설치된 환경에서만 동작.
미설치 시 명확한 에러로 안내.

핵심 흐름:
  1. 모델 + 토크나이저 로드 (양자화 적용)
  2. PEFT/LoRA 어댑터 적용
  3. SFTTrainer로 학습 실행
  4. 체크포인트 저장 + 손실 값 기록
  5. WebSocket 이벤트 발행 (콜백 기반)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobSpec, JobState, Quantization, TrainConfig, TrainStrategy

log = get_logger(__name__)

# 진행 이벤트 콜백 타입 (job_id, stage, msg, extra)
ProgressCallback = Callable[[str, int, str, dict[str, Any]], None]


class PipelineTrainer:
    """
    Stage 7: 실제 학습 실행기.

    책임:
    - 모델/토크나이저 로드 및 양자화 설정
    - PEFT LoRA 어댑터 주입
    - SFTTrainer 설정 및 실행
    - 체크포인트 관리
    - 진행 상황 콜백 발행
    - JobSpec.train_loss / eval_loss / checkpoint_dir 갱신
    """

    def __init__(self, progress_cb: ProgressCallback | None = None) -> None:
        # 진행 상황을 실시간으로 전달할 콜백 (WebSocket 전송 등)
        self._progress_cb = progress_cb or self._default_progress_cb

    def train(self, spec: JobSpec) -> str:
        """
        학습 실행.

        Parameters
        ----------
        spec:
            Stage 6까지 채워진 JobSpec.
            model_local_path, dataset_path, train_config 필요.

        Returns
        -------
        str
            체크포인트가 저장된 디렉토리 경로.

        Raises
        ------
        WFPError(WFP-0302)
            torch/transformers 미설치.
        WFPError(WFP-0701)
            CUDA OOM 발생.
        WFPError(WFP-0702)
            기타 학습 중 크래시.
        """
        config = spec.train_config
        if not config:
            raise WFPError(
                code=WFPErrorCode.CONFIG_BUILD_FAILED,
                message="train_config가 없습니다. Stage 6을 먼저 실행하세요.",
                stage=7,
            )

        # 학습 라이브러리 임포트 확인
        self._check_deps()

        self._emit(spec, "학습 시작")
        spec.update_stage(stage=7, state=JobState.RUNNING)

        try:
            model, tokenizer = self._load_model(spec, config)
            model = self._apply_peft(model, config)
            trainer = self._build_trainer(spec, model, tokenizer, config)

            start_time = time.time()
            self._emit(spec, "SFTTrainer 학습 실행 중...")

            train_result = trainer.train()

            elapsed = time.time() - start_time
            spec.train_loss = train_result.training_loss

            # 최종 체크포인트 저장
            trainer.save_model(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)

            spec.checkpoint_dir = config.output_dir

            # eval loss (eval dataset이 있을 경우)
            try:
                eval_result = trainer.evaluate()
                spec.eval_loss = eval_result.get("eval_loss")
            except Exception:
                pass

            spec.update_stage(stage=7, state=JobState.RUNNING)

            self._emit(
                spec,
                f"학습 완료 — {elapsed:.0f}초 소요",
                extra={"train_loss": spec.train_loss, "eval_loss": spec.eval_loss},
            )

            log.info(
                "Stage 7: 학습 완료",
                extra={
                    "stage": 7,
                    "job_id": str(spec.job_id),
                    "extra": {
                        "train_loss": spec.train_loss,
                        "elapsed_sec": round(elapsed, 1),
                        "checkpoint": config.output_dir,
                    },
                },
            )

            return config.output_dir

        except RuntimeError as e:
            err_str = str(e)
            if "out of memory" in err_str.lower() or "CUDA out of memory" in err_str:
                raise WFPError(
                    code=WFPErrorCode.TRAIN_CUDA_OOM,
                    message="GPU 메모리 부족 (OOM). batch_size나 max_seq_length를 줄이세요.",
                    stage=7,
                    detail=err_str,
                ) from e
            raise WFPError(
                code=WFPErrorCode.TRAIN_CRASHED,
                message=f"학습 중 오류 발생: {err_str[:200]}",
                stage=7,
                detail=err_str,
            ) from e

    # ─────────────────────────────────────────
    # 내부 구현
    # ─────────────────────────────────────────

    @staticmethod
    def _check_deps() -> None:
        """학습 필수 의존성 확인."""
        missing = []
        for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise WFPError(
                code=WFPErrorCode.MISSING_DEPENDENCY,
                message=f"학습 패키지 미설치: {missing}",
                stage=7,
                detail="pip install wfp-worker[train] 또는 uv sync --extra train",
            )

    @staticmethod
    def _load_model(spec: JobSpec, config: TrainConfig):  # type: ignore[return]
        """모델과 토크나이저 로드 (양자화 포함)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = spec.model_local_path or spec.model_hf_repo
        if not model_path:
            raise WFPError(
                code=WFPErrorCode.MODEL_NOT_FOUND,
                message="model_local_path가 비어 있습니다.",
                stage=7,
            )

        # 양자화 설정
        bnb_config = None
        if config.quantization == Quantization.INT4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.quantization == Quantization.INT8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # 디바이스 맵 결정
        device_map: str | dict
        if config.strategy in (TrainStrategy.QLORA, TrainStrategy.LORA):
            device_map = "auto"
        else:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",  # SFT 학습에 right padding 권장
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    @staticmethod
    def _apply_peft(model, config: TrainConfig):  # type: ignore[return]
        """LoRA/QLoRA PEFT 어댑터 적용."""
        if config.strategy not in (TrainStrategy.LORA, TrainStrategy.QLORA):
            return model

        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        # QLoRA: kbit 학습 준비
        if config.quantization in (Quantization.INT4, Quantization.INT8):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )

        lora_cfg = config.lora
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        return get_peft_model(model, peft_config)

    def _build_trainer(
        self,
        spec: JobSpec,
        model,
        tokenizer,
        config: TrainConfig,
    ):  # type: ignore[return]
        """SFTTrainer 인스턴스 생성."""
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        dataset_path = spec.dataset_path
        if not dataset_path:
            raise WFPError(
                code=WFPErrorCode.DATA_DIR_NOT_FOUND,
                message="dataset_path가 없습니다. Stage 5를 먼저 실행하세요.",
                stage=7,
            )

        # JSONL 형식 데이터 로드
        train_file = str(Path(dataset_path) / "train.jsonl")
        val_file = str(Path(dataset_path) / "val.jsonl")

        dataset = load_dataset(
            "json",
            data_files={"train": train_file, "validation": val_file},
        )

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            logging_steps=config.logging_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=["tensorboard"],
        )

        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            max_seq_length=config.max_seq_length,
            dataset_text_field="instruction",
            packing=False,
            callbacks=[self._make_progress_callback(spec)],
        )

    def _make_progress_callback(self, spec: JobSpec):
        """학습 진행 콜백 → ProgressCallback 변환."""
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        cb = self._progress_cb

        class _WFPCallback(TrainerCallback):
            def on_log(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                logs: dict | None = None,
                **kwargs: Any,
            ) -> None:
                if logs:
                    cb(
                        str(spec.job_id),
                        7,
                        f"Step {state.global_step}",
                        {
                            "loss": logs.get("loss"),
                            "eval_loss": logs.get("eval_loss"),
                            "step": state.global_step,
                            "epoch": logs.get("epoch"),
                        },
                    )

        return _WFPCallback()

    def _emit(
        self, spec: JobSpec, msg: str, extra: dict[str, Any] | None = None
    ) -> None:
        """진행 이벤트 발행 헬퍼."""
        self._progress_cb(str(spec.job_id), 7, msg, extra or {})

    @staticmethod
    def _default_progress_cb(
        job_id: str, stage: int, msg: str, extra: dict[str, Any]
    ) -> None:
        """콜백 미제공 시 단순 로그 출력."""
        log.info(msg, extra={"stage": stage, "job_id": job_id, "extra": extra})
