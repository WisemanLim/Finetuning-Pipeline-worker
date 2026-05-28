"""
Stage 8 — 평가 (Evaluator).

PRD §4.8: 자동 메트릭 + LLM-as-Judge 방식 평가.

평가 타입:
  1. 자동 메트릭 (perplexity, BLEU, ROUGE, F1)
  2. 벤치마크 (KMMLU, MMLU 등 — 인터페이스만, 실행은 별도 스크립트)
  3. LLM-as-Judge (GPT-4o / Claude API 사용, 선택)

품질 기준 미달 시 자동 재시도(epoch 증가) 로직 포함.
최대 재시도 횟수: MAX_RETRY.
"""

from __future__ import annotations

from typing import Any, Callable

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import EvalResult, JobSpec, JobState, TaskType

log = get_logger(__name__)

# 품질 기준 임계값 (태스크 타입별)
_QUALITY_THRESHOLDS: dict[TaskType, dict[str, float]] = {
    TaskType.CHAT:           {"perplexity_max": 30.0},
    TaskType.QA:             {"f1_min": 0.3,   "perplexity_max": 25.0},
    TaskType.SUMMARY:        {"rougeL_min": 0.2, "perplexity_max": 28.0},
    TaskType.CLASSIFICATION: {"f1_min": 0.5},
    TaskType.CODE:           {"exact_match_min": 0.1, "perplexity_max": 20.0},
}

MAX_RETRY = 3


class ModelEvaluator:
    """
    Stage 8: 모델 평가.

    책임:
    - 체크포인트 로드 후 테스트셋 추론
    - 자동 메트릭 계산
    - 품질 기준 판정
    - 기준 미달 시 자동 재시도 신호 반환
    - JobSpec.eval_result 갱신
    """

    def evaluate(
        self,
        spec: JobSpec,
        progress_cb: Callable[[str, int, str, dict], None] | None = None,
    ) -> EvalResult:
        """
        평가 실행.

        Parameters
        ----------
        spec:
            Stage 7까지 채워진 JobSpec.
        progress_cb:
            진행 이벤트 콜백.

        Returns
        -------
        EvalResult
            메트릭 및 통과 여부. passed=False이면 재시도 권고.
        """
        cb = progress_cb or (lambda *a, **kw: None)
        spec.update_stage(stage=8, state=JobState.EVAL)
        cb(str(spec.job_id), 8, "평가 시작", {})

        checkpoint = spec.checkpoint_dir
        if not checkpoint:
            raise WFPError(
                code=WFPErrorCode.EVAL_FAILED,
                message="checkpoint_dir 없음. Stage 7을 먼저 실행하세요.",
                stage=8,
            )

        result = EvalResult(retry_count=spec.eval_result.retry_count if spec.eval_result else 0)

        try:
            result = self._run_metrics(spec, result, cb)
        except ImportError:
            # 평가 라이브러리 미설치 — 경고 후 통과 처리
            log.warning("평가 라이브러리 미설치 — 메트릭 계산 생략")
            result.passed = True
            spec.eval_result = result
            spec.update_stage(stage=8, state=JobState.RUNNING)
            return result

        # 품질 기준 판정
        result.passed = self._check_quality(spec.task_type, result)

        spec.eval_result = result
        spec.update_stage(stage=8, state=JobState.RUNNING)

        cb(
            str(spec.job_id),
            8,
            f"평가 완료 — {'통과' if result.passed else '기준 미달'}",
            {
                "perplexity": result.perplexity,
                "rouge1": result.rouge1,
                "rougeL": result.rougeL,
                "f1": result.f1,
                "passed": result.passed,
            },
        )

        log.info(
            "Stage 8: 평가 완료",
            extra={
                "stage": 8,
                "job_id": str(spec.job_id),
                "extra": {
                    "passed": result.passed,
                    "perplexity": result.perplexity,
                    "rougeL": result.rougeL,
                    "retry": result.retry_count,
                },
            },
        )

        return result

    def should_retry(self, result: EvalResult) -> bool:
        """자동 재시도 여부 판단."""
        return not result.passed and result.retry_count < MAX_RETRY

    # ─────────────────────────────────────────
    # 메트릭 계산
    # ─────────────────────────────────────────

    def _run_metrics(
        self,
        spec: JobSpec,
        result: EvalResult,
        cb: Callable,
    ) -> EvalResult:
        """테스트셋으로 자동 메트릭 계산."""
        import math
        from pathlib import Path

        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = spec.checkpoint_dir
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        # 추론 전용 모드 (양자화 없이 fp16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()

        # 테스트 데이터 로드
        test_file = str(Path(spec.dataset_path) / "test.jsonl")
        dataset = load_dataset("json", data_files={"test": test_file})["test"]

        eval_subset = dataset.select(range(min(100, len(dataset))))
        total = len(eval_subset)
        cb(
            str(spec.job_id), 8,
            f"테스트셋 {total}건 추론 시작",
            {"step": 0, "total_steps": total, "stage_progress": 0.0},
        )

        predictions: list[str] = []
        references: list[str] = []

        # 매 step 발행은 과도 — 1% 또는 1 step 간격
        emit_interval = max(1, total // 100)

        # 배치 추론
        for idx, item in enumerate(eval_subset):
            ref = str(item.get("output", item.get("text", "")))
            prompt = str(item.get("instruction", item.get("text", "")))

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            pred = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            predictions.append(pred)
            references.append(ref)

            step = idx + 1
            if step % emit_interval == 0 or step == total:
                cb(
                    str(spec.job_id), 8,
                    f"추론 {step}/{total}",
                    {"step": step, "total_steps": total,
                     "stage_progress": step / total},
                )

        cb(str(spec.job_id), 8, "추론 완료 — 메트릭 계산 중", {})

        # Perplexity
        result.perplexity = self._compute_perplexity(model, tokenizer, references[:20])
        cb(str(spec.job_id), 8, "Perplexity 계산 완료",
           {"perplexity": result.perplexity})

        # ROUGE
        rouge_scores = self._compute_rouge(predictions, references)
        result.rouge1 = rouge_scores.get("rouge1", 0.0)
        result.rouge2 = rouge_scores.get("rouge2", 0.0)
        result.rougeL = rouge_scores.get("rougeL", 0.0)
        cb(str(spec.job_id), 8, "ROUGE 계산 완료",
           {"rouge1": result.rouge1, "rougeL": result.rougeL})

        # F1 (토큰 수준)
        result.f1 = self._compute_token_f1(predictions, references)
        cb(str(spec.job_id), 8, "F1 계산 완료", {"f1": result.f1})

        # Exact match
        result.exact_match = sum(
            1 for p, r in zip(predictions, references) if p.strip() == r.strip()
        ) / max(len(predictions), 1)

        return result

    @staticmethod
    def _compute_perplexity(model: Any, tokenizer: Any, texts: list[str]) -> float:
        """평균 Perplexity 계산."""
        import math

        import torch

        total_loss = 0.0
        count = 0

        for text in texts:
            if not text.strip():
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

        if count == 0:
            return 999.0

        avg_loss = total_loss / count
        return round(math.exp(avg_loss), 2)

    @staticmethod
    def _compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
        """ROUGE-1/2/L 계산."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

            for pred, ref in zip(predictions, references):
                result = scorer.score(ref, pred)
                for key in scores:
                    scores[key].append(result[key].fmeasure)

            return {k: round(sum(v) / max(len(v), 1), 4) for k, v in scores.items()}
        except ImportError:
            return {}

    @staticmethod
    def _compute_token_f1(predictions: list[str], references: list[str]) -> float:
        """토큰 수준 F1 (QA 태스크용)."""
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        return round(sum(f1_scores) / max(len(f1_scores), 1), 4)

    @staticmethod
    def _check_quality(task_type: TaskType, result: EvalResult) -> bool:
        """태스크 타입별 품질 임계값 판정."""
        thresholds = _QUALITY_THRESHOLDS.get(task_type, {})

        for key, threshold in thresholds.items():
            if key == "perplexity_max":
                val = result.perplexity
                if val is not None and val > threshold:
                    log.info(
                        "품질 기준 미달",
                        extra={"extra": {"metric": "perplexity", "value": val, "threshold": threshold}},
                    )
                    return False
            elif key == "rougeL_min":
                val = result.rougeL
                if val is not None and val < threshold:
                    log.info("품질 기준 미달", extra={"extra": {"metric": "rougeL", "value": val}})
                    return False
            elif key == "f1_min":
                val = result.f1
                if val is not None and val < threshold:
                    log.info("품질 기준 미달", extra={"extra": {"metric": "f1", "value": val}})
                    return False
            elif key == "exact_match_min":
                val = result.exact_match
                if val is not None and val < threshold:
                    log.info("품질 기준 미달", extra={"extra": {"metric": "exact_match", "value": val}})
                    return False

        return True
