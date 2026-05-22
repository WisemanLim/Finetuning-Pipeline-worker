"""
Stage 6 — 학습 설정 자동 생성 테스트.

시나리오:
  S6-01  정상 빌드 → TrainConfig 반환
  S6-02  LoRA 전략 → LoRAConfig 포함
  S6-03  QLoRA 전략 → quantization=INT4
  S6-04  데이터 100건 → epochs=3
  S6-05  데이터 200건 미만 → epochs=5
  S6-06  사용자 오버라이드 적용 (lr, epochs)
  S6-07  알 수 없는 오버라이드 키 무시 (예외 없음)
  S6-08  output_dir 생성 확인
  S6-09  spec.train_config 갱신 확인
  S6-10  리소스 추정 dict 반환 확인
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from wfp_worker.common.types import Quantization, TrainStrategy
from wfp_worker.stage6_train_config import TrainConfigBuilder


class TestTrainConfigBuilder:
    def test_s6_01_returns_train_config(self, full_spec):
        """S6-01: build() → TrainConfig 반환."""
        config = TrainConfigBuilder().build(full_spec)
        assert config is not None
        assert config.strategy is not None

    def test_s6_02_lora_config_included(self, full_spec):
        """S6-02: LoRA/QLoRA 전략이면 LoRAConfig 포함."""
        full_spec.recommended_plan.recommended_strategy = TrainStrategy.QLORA
        config = TrainConfigBuilder().build(full_spec)
        assert config.lora is not None
        assert config.lora.r > 0

    def test_s6_03_qlora_quantization(self, full_spec):
        """S6-03: QLoRA 전략 → INT4 양자화."""
        full_spec.recommended_plan.recommended_strategy = TrainStrategy.QLORA
        full_spec.recommended_plan.recommended_quant = Quantization.INT4
        config = TrainConfigBuilder().build(full_spec)
        assert config.quantization == Quantization.INT4

    def test_s6_04_epochs_small_dataset(self, full_spec):
        """S6-04: 500건 미만 → 5 에폭 (소량 데이터 다회 학습)."""
        full_spec.num_train = 100
        config = TrainConfigBuilder().build(full_spec)
        assert config.num_epochs == 5

    def test_s6_05_epochs_medium_dataset(self, full_spec):
        """S6-05: 500건 이상 5000건 미만 → 3 에폭."""
        full_spec.num_train = 1000
        config = TrainConfigBuilder().build(full_spec)
        assert config.num_epochs == 3

    def test_s6_06_override_applied(self, full_spec):
        """S6-06: 오버라이드 적용."""
        config = TrainConfigBuilder().build(
            full_spec,
            override={"learning_rate": 1e-5, "num_epochs": 10},
        )
        assert abs(config.learning_rate - 1e-5) < 1e-10
        assert config.num_epochs == 10

    def test_s6_07_unknown_override_ignored(self, full_spec):
        """S6-07: 알 수 없는 오버라이드 키 → 예외 없음."""
        config = TrainConfigBuilder().build(
            full_spec,
            override={"nonexistent_key": "value"},
        )
        assert config is not None

    def test_s6_08_output_dir_created(self, full_spec, tmp_path):
        """S6-08: output_dir 생성."""
        # 체크포인트 루트를 tmp_path로 임시 변경
        with patch(
            "wfp_worker.stage6_train_config.config_builder._DEFAULT_CKPT_ROOT",
            tmp_path / "ckpts",
        ):
            config = TrainConfigBuilder().build(full_spec)
        assert os.path.exists(config.output_dir)

    def test_s6_09_spec_config_updated(self, full_spec):
        """S6-09: build 후 spec.train_config 갱신."""
        TrainConfigBuilder().build(full_spec)
        assert full_spec.train_config is not None

    def test_s6_10_resource_estimate(self, full_spec):
        """S6-10: estimate_resources() dict 반환."""
        builder = TrainConfigBuilder()
        builder.build(full_spec)
        est = builder.estimate_resources(full_spec)
        assert "estimated_hours" in est
        assert "estimated_vram_gb" in est
        assert "estimated_disk_gb" in est
