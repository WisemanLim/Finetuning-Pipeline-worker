"""
Stage 3 — 환경 설정 테스트.

시나리오:
  S3-01  현재 Python 버전 >= 3.12 통과
  S3-02  필수 패키지(psutil, pydantic 등) 정상 임포트
  S3-03  env_ready=True 후 JobSpec 갱신 확인
  S3-04  python_version 필드 채워짐
  S3-05  GPU 없는 환경에서 경고 메시지 반환
  S3-06  NVIDIA GPU 있으나 CUDA 미활성 → driver warning
  S3-07  Apple Silicon 감지 시 macOS 권고 메시지
  S3-08  Windows 환경 플래그 → WSL2 권고 (시뮬레이션)
  S3-09  stage 3으로 전이 확인
  S3-10  env_ready=True 시 is_ready 속성 True
"""

from unittest.mock import patch

import pytest

from wfp_worker.common.types import CPUInfo, GPUInfo, HWCaps
from wfp_worker.stage3_env_setup import EnvManager


class TestEnvManager:
    def test_s3_01_python_version_ok(self, minimal_spec):
        """S3-01: 현재 Python >= 3.12."""
        result = EnvManager().setup(minimal_spec)
        assert result.python_ok is True

    def test_s3_02_required_packages_installed(self, minimal_spec):
        """S3-02: 필수 패키지 누락 없음."""
        result = EnvManager().setup(minimal_spec)
        assert result.missing_required == []

    def test_s3_03_spec_env_ready_updated(self, minimal_spec):
        """S3-03: spec.env_ready 갱신."""
        EnvManager().setup(minimal_spec)
        assert minimal_spec.env_ready is True

    def test_s3_04_python_version_filled(self, minimal_spec):
        """S3-04: python_version 필드 채워짐."""
        EnvManager().setup(minimal_spec)
        assert "." in minimal_spec.python_version  # "3.12.x" 형식

    def test_s3_05_no_gpu_warning(self, minimal_spec, hw_caps_cpu_only):
        """S3-05: GPU 없는 환경 → driver warning."""
        minimal_spec.hw_caps = hw_caps_cpu_only
        result = EnvManager().setup(minimal_spec)
        assert any("GPU" in w for w in result.driver_warnings)

    def test_s3_06_nvidia_no_cuda_warning(self, minimal_spec):
        """S3-06: NVIDIA GPU 있으나 CUDA 미활성 → 경고."""
        caps = HWCaps(
            os="linux",
            cpu=CPUInfo(),
            gpus=[GPUInfo(vendor="nvidia", vram_gb=8.0, name="RTX 3060")],
            ram_gb=16.0,
            disk_free_gb=50.0,
            max_vram_gb=8.0,
            cuda_available=False,  # CUDA 비활성
        )
        minimal_spec.hw_caps = caps
        result = EnvManager().setup(minimal_spec)
        assert any("CUDA" in w for w in result.driver_warnings)

    def test_s3_07_apple_silicon_os_warning(self, minimal_spec, hw_caps_apple_silicon):
        """S3-07: Apple Silicon → macOS 권고 메시지."""
        minimal_spec.hw_caps = hw_caps_apple_silicon
        result = EnvManager().setup(minimal_spec)
        assert any("macOS" in w or "bitsandbytes" in w for w in result.os_warnings)

    def test_s3_08_windows_wsl_warning(self, minimal_spec):
        """S3-08: Windows + 비NVIDIA → WSL2 권고."""
        caps = HWCaps(
            os="windows",
            cpu=CPUInfo(),
            gpus=[],
            ram_gb=16.0,
            disk_free_gb=50.0,
            max_vram_gb=0.0,
        )
        minimal_spec.hw_caps = caps
        result = EnvManager().setup(minimal_spec)
        assert any("WSL" in w for w in result.os_warnings)

    def test_s3_09_stage_transition(self, minimal_spec):
        """S3-09: setup 후 current_stage == 3."""
        EnvManager().setup(minimal_spec)
        assert minimal_spec.current_stage == 3

    def test_s3_10_is_ready_property(self, minimal_spec):
        """S3-10: env_ready=True이면 EnvCheckResult.is_ready=True."""
        result = EnvManager().setup(minimal_spec)
        assert result.is_ready is True
