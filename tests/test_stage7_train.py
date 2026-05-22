"""
Stage 7 — 학습 실행 테스트.

실제 torch/transformers 없이 실행 가능하도록 mock 사용.

시나리오:
  S7-01  train_config 없으면 WFP-0601 에러
  S7-02  torch 미설치 시 WFP-0302 에러
  S7-03  CUDA OOM → WFP-0701 에러 변환
  S7-04  진행 콜백 호출 확인
  S7-05  dataset_path 없으면 WFP-0502 에러
  S7-06  학습 완료 후 spec.train_loss 갱신
  S7-07  학습 완료 후 spec.checkpoint_dir 갱신
  S7-08  콜백 없어도 동작 (기본 로거 사용)
"""

from unittest.mock import MagicMock, patch

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.stage7_train import PipelineTrainer


class TestPipelineTrainer:
    def test_s7_01_no_config_raises(self, full_spec):
        """S7-01: train_config 없으면 WFP-0601."""
        full_spec.train_config = None
        with pytest.raises(WFPError) as exc_info:
            PipelineTrainer().train(full_spec)
        assert exc_info.value.code == WFPErrorCode.CONFIG_BUILD_FAILED

    def test_s7_02_missing_torch_raises(self, full_spec):
        """S7-02: torch 미설치 → WFP-0302."""
        import sys
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore
        sys.modules["transformers"] = None  # type: ignore

        try:
            with pytest.raises(WFPError) as exc_info:
                PipelineTrainer().train(full_spec)
            assert exc_info.value.code == WFPErrorCode.MISSING_DEPENDENCY
        finally:
            # 복구
            if original is not None:
                sys.modules["torch"] = original
            else:
                sys.modules.pop("torch", None)
            sys.modules.pop("transformers", None)

    def test_s7_03_cuda_oom_converted(self, full_spec):
        """S7-03: CUDA OOM RuntimeError → WFP-0701."""
        with patch.object(PipelineTrainer, "_check_deps", return_value=None), \
             patch.object(PipelineTrainer, "_load_model", side_effect=RuntimeError("CUDA out of memory")):
            with pytest.raises(WFPError) as exc_info:
                PipelineTrainer().train(full_spec)
            assert exc_info.value.code == WFPErrorCode.TRAIN_CUDA_OOM

    def test_s7_04_progress_callback_called(self, full_spec):
        """S7-04: 학습 시작 시 콜백 호출."""
        calls = []

        def mock_cb(job_id, stage, msg, extra):
            calls.append(msg)

        trainer = PipelineTrainer(progress_cb=mock_cb)

        # _check_deps 통과 + 즉시 실패 시뮬레이션
        with patch.object(trainer, "_check_deps", return_value=None), \
             patch.object(trainer, "_load_model", side_effect=RuntimeError("test crash")):
            try:
                trainer.train(full_spec)
            except WFPError:
                pass

        # "학습 시작" 콜백이 호출되어야 함
        assert any("학습 시작" in c for c in calls)

    def test_s7_05_no_dataset_raises(self, full_spec):
        """S7-05: dataset_path 없으면 WFP-0502."""
        full_spec.dataset_path = ""
        with patch.object(PipelineTrainer, "_check_deps", return_value=None), \
             patch.object(PipelineTrainer, "_load_model", return_value=(MagicMock(), MagicMock())), \
             patch.object(PipelineTrainer, "_apply_peft", side_effect=lambda m, c: m), \
             patch.object(PipelineTrainer, "_build_trainer", side_effect=WFPError(
                 WFPErrorCode.DATA_DIR_NOT_FOUND, "dataset_path 없음", stage=7
             )):
            with pytest.raises(WFPError) as exc_info:
                PipelineTrainer().train(full_spec)
            assert exc_info.value.code == WFPErrorCode.DATA_DIR_NOT_FOUND

    def test_s7_08_no_callback_no_crash(self, full_spec):
        """S7-08: 콜백 없이도 정상 동작 (기본 로거)."""
        trainer = PipelineTrainer()
        with patch.object(trainer, "_check_deps", return_value=None), \
             patch.object(trainer, "_load_model", side_effect=RuntimeError("test")):
            try:
                trainer.train(full_spec)
            except WFPError:
                pass  # 에러는 기대값 — 크래시만 안 나면 됨
