"""
Stage 2 — HW 프로빙 테스트.

시나리오:
  S2-01  HWProber.probe() 실행 → HWCaps 반환 (OS, RAM, CPU)
  S2-02  CPU/RAM 값 0 이상 검증
  S2-03  PlanRecommender — NVIDIA 16GB → qlora/int4 플랜
  S2-04  PlanRecommender — Apple Silicon → mps 백엔드 플랜
  S2-05  PlanRecommender — CPU 전용 → 경고 포함 플랜
  S2-06  규칙 파일 없을 때 내장 폴백 사용
  S2-07  os 필드가 linux/windows/darwin 중 하나
  S2-08  멀티 GPU 환경에서 max_vram_gb = 단일 GPU 최대값
  S2-09  플랜 max_model_params_b > 0 (CPU 전용 제외)
  S2-10  플랜 recommended_strategy 가 TrainStrategy 열거형
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from wfp_worker.common.types import HWCaps, TrainStrategy
from wfp_worker.stage2_hw_probe import HWProber, PlanRecommender


class TestHWProber:
    def test_s2_01_returns_hw_caps(self):
        """S2-01: probe() 실행 — HWCaps 인스턴스 반환."""
        caps = HWProber().probe()
        assert isinstance(caps, HWCaps)

    def test_s2_02_ram_positive(self):
        """S2-02: RAM > 0."""
        caps = HWProber().probe()
        assert caps.ram_gb > 0

    def test_s2_02_cpu_cores_positive(self):
        """S2-02: CPU 코어 수 > 0."""
        caps = HWProber().probe()
        assert caps.cpu.cores_logical > 0

    def test_s2_07_os_valid(self):
        """S2-07: os 필드가 유효한 플랫폼 문자열."""
        caps = HWProber().probe()
        assert caps.os in ("linux", "windows", "darwin")

    def test_s2_08_max_vram_consistent(self):
        """S2-08: max_vram_gb = max(gpus.vram_gb)."""
        caps = HWProber().probe()
        if caps.gpus:
            expected = max(g.vram_gb for g in caps.gpus)
            assert abs(caps.max_vram_gb - expected) < 0.1
        else:
            assert caps.max_vram_gb == 0.0


class TestPlanRecommender:
    def test_s2_03_nvidia_16gb_plan(self, hw_caps_nvidia_16gb: HWCaps):
        """S2-03: NVIDIA 16GB → QLoRA/int4 플랜."""
        plan = PlanRecommender().recommend(hw_caps_nvidia_16gb)
        assert plan.recommended_strategy == TrainStrategy.QLORA
        assert plan.backend == "cuda"
        assert plan.max_model_params_b > 0

    def test_s2_04_apple_silicon_plan(self, hw_caps_apple_silicon: HWCaps):
        """S2-04: Apple Silicon → mps 백엔드."""
        plan = PlanRecommender().recommend(hw_caps_apple_silicon)
        assert plan.backend == "mps"

    def test_s2_05_cpu_only_warning(self, hw_caps_cpu_only: HWCaps):
        """S2-05: CPU 전용 → warning 필드 비어 있지 않음."""
        plan = PlanRecommender().recommend(hw_caps_cpu_only)
        assert plan.backend == "cpu"
        assert len(plan.warning) > 0

    def test_s2_06_fallback_no_rules_file(self, hw_caps_nvidia_16gb: HWCaps):
        """S2-06: 규칙 파일 없으면 내장 폴백 사용."""
        recommender = PlanRecommender(rules_path=Path("/nonexistent/path.yaml"))
        plan = recommender.recommend(hw_caps_nvidia_16gb)
        assert plan.recommended_strategy is not None

    def test_s2_09_params_positive_for_gpu(self, hw_caps_nvidia_16gb: HWCaps):
        """S2-09: GPU 환경에서 max_model_params_b > 0."""
        plan = PlanRecommender().recommend(hw_caps_nvidia_16gb)
        assert plan.max_model_params_b > 0

    def test_s2_10_strategy_is_enum(self, hw_caps_nvidia_16gb: HWCaps):
        """S2-10: recommended_strategy가 TrainStrategy 열거형."""
        plan = PlanRecommender().recommend(hw_caps_nvidia_16gb)
        assert isinstance(plan.recommended_strategy, TrainStrategy)
