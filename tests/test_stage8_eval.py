"""
Stage 8 — 평가 테스트.

시나리오:
  S8-01  checkpoint_dir 없으면 WFP-0801
  S8-02  평가 라이브러리 미설치 시 passed=True 폴백
  S8-03  perplexity 임계값 초과 → passed=False
  S8-04  ROUGE-L 임계값 초과 → passed=True
  S8-05  token F1 계산 정확성
  S8-06  should_retry — passed=False이고 retry<MAX → True
  S8-07  should_retry — MAX_RETRY 초과 시 False
  S8-08  should_retry — passed=True 이면 False
  S8-09  JobSpec.eval_result 갱신 확인
  S8-10  JobSpec.state = EVAL 전이 확인
"""

from unittest.mock import MagicMock, patch

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.types import EvalResult, JobState, TaskType
from wfp_worker.stage8_eval import ModelEvaluator
from wfp_worker.stage8_eval.evaluator import MAX_RETRY


class TestModelEvaluator:
    def test_s8_01_no_checkpoint_raises(self, full_spec):
        """S8-01: checkpoint_dir 없으면 WFP-0801."""
        full_spec.checkpoint_dir = ""
        with pytest.raises(WFPError) as exc_info:
            ModelEvaluator().evaluate(full_spec)
        assert exc_info.value.code == WFPErrorCode.EVAL_FAILED

    def test_s8_02_missing_lib_fallback(self, full_spec, tmp_path):
        """S8-02: 평가 라이브러리 없으면 passed=True 폴백."""
        full_spec.checkpoint_dir = str(tmp_path / "ckpt")
        (tmp_path / "ckpt").mkdir()

        evaluator = ModelEvaluator()
        # _run_metrics가 ImportError 발생하도록 mock
        with patch.object(evaluator, "_run_metrics", side_effect=ImportError("no torch")):
            result = evaluator.evaluate(full_spec)

        assert result.passed is True

    def test_s8_03_high_perplexity_fails(self):
        """S8-03: perplexity > 임계값 → passed=False."""
        evaluator = ModelEvaluator()
        result = EvalResult(perplexity=100.0)
        passed = evaluator._check_quality(TaskType.CHAT, result)
        assert passed is False

    def test_s8_04_good_rouge_passes(self):
        """S8-04: rougeL >= 임계값 → passed=True."""
        evaluator = ModelEvaluator()
        result = EvalResult(rougeL=0.5, perplexity=10.0)
        passed = evaluator._check_quality(TaskType.SUMMARY, result)
        assert passed is True

    def test_s8_05_token_f1_calculation(self):
        """S8-05: token F1 계산."""
        evaluator = ModelEvaluator()
        preds = ["파이썬 코드 작성"]
        refs = ["파이썬 코드 실행"]
        f1 = evaluator._compute_token_f1(preds, refs)
        # "파이썬"과 "코드" 공통 → F1 > 0
        assert f1 > 0.0

    def test_s8_06_should_retry_true(self):
        """S8-06: 기준 미달 + retry < MAX → should_retry=True."""
        evaluator = ModelEvaluator()
        result = EvalResult(passed=False, retry_count=0)
        assert evaluator.should_retry(result) is True

    def test_s8_07_should_retry_max_exceeded(self):
        """S8-07: retry >= MAX → should_retry=False."""
        evaluator = ModelEvaluator()
        result = EvalResult(passed=False, retry_count=MAX_RETRY)
        assert evaluator.should_retry(result) is False

    def test_s8_08_should_retry_when_passed(self):
        """S8-08: passed=True → should_retry=False."""
        evaluator = ModelEvaluator()
        result = EvalResult(passed=True, retry_count=0)
        assert evaluator.should_retry(result) is False

    def test_s8_09_spec_eval_result_updated(self, full_spec, tmp_path):
        """S8-09: evaluate 후 spec.eval_result 갱신."""
        full_spec.checkpoint_dir = str(tmp_path / "ckpt")
        (tmp_path / "ckpt").mkdir()

        evaluator = ModelEvaluator()
        with patch.object(evaluator, "_run_metrics", side_effect=ImportError):
            evaluator.evaluate(full_spec)

        assert full_spec.eval_result is not None

    def test_s8_10_state_transitions_to_eval(self, full_spec, tmp_path):
        """S8-10: evaluate 호출 시 state=EVAL 전이 (일시적)."""
        full_spec.checkpoint_dir = str(tmp_path / "ckpt")
        (tmp_path / "ckpt").mkdir()

        states = []

        def mock_run_metrics(spec, result, cb):
            states.append(spec.state)
            raise ImportError  # 즉시 폴백 트리거

        evaluator = ModelEvaluator()
        with patch.object(evaluator, "_run_metrics", side_effect=mock_run_metrics):
            evaluator.evaluate(full_spec)

        assert JobState.EVAL in states
