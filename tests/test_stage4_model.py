"""
Stage 4 — 모델 선택 및 다운로드 테스트.

시나리오:
  S4-01  ModelCatalog — J 섹션 코드 모델 우선 반환
  S4-02  ModelCatalog — VRAM 필터 (6GB 환경에서 대형 모델 제외)
  S4-03  ModelCatalog — 최대 결과 수 제한
  S4-04  ModelPicker — 자동 선택 시 1순위 채택
  S4-05  ModelPicker — 후보 없으면 WFP-0402
  S4-06  ModelPicker — manual_repo 지정 시 해당 모델 사용
  S4-07  ModelPicker — 카탈로그 외 manual_repo 허용 (경고만)
  S4-08  ModelPicker — spec.model_hf_repo 갱신 확인
  S4-09  ModelDownloader — 이미 캐시된 경우 재다운로드 없이 경로 반환
  S4-10  ModelDownloader — 디스크 부족 시 WFP-0404
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.stage4_model import ModelCatalog, ModelDownloader, ModelPicker


class TestModelCatalog:
    def test_s4_01_code_model_priority_for_j(self, minimal_spec):
        """S4-01: J 섹션에서 코드 모델 우선 반환."""
        catalog = ModelCatalog()
        candidates = catalog.get_candidates(ksic_section="J", max_vram_gb=16.0, language="ko")
        assert len(candidates) > 0
        # 상위 후보 중 code 태그 존재 확인
        top_tags = [t for c in candidates[:3] for t in c.tags]
        assert "code" in top_tags

    def test_s4_02_vram_filter(self):
        """S4-02: 6GB 환경에서 대형 모델 제외."""
        catalog = ModelCatalog()
        candidates = catalog.get_candidates(ksic_section="J", max_vram_gb=6.0)
        for c in candidates:
            assert c.hw_min_vram_gb <= 6.0

    def test_s4_03_max_results(self):
        """S4-03: max_results 제한 준수."""
        catalog = ModelCatalog()
        candidates = catalog.get_candidates(ksic_section="J", max_vram_gb=80.0, max_results=3)
        assert len(candidates) <= 3


class TestModelPicker:
    def test_s4_04_auto_pick_first(self, full_spec):
        """S4-04: 자동 선택 시 1순위 채택."""
        picker = ModelPicker()
        repo = picker.pick(full_spec)
        assert isinstance(repo, str)
        assert "/" in repo  # "org/model" 형식

    def test_s4_05_no_candidates_raises(self, full_spec):
        """S4-05: VRAM 0이고 max_params=0 → WFP-0402."""
        full_spec.recommended_plan.max_model_params_b = 0.0
        full_spec.hw_caps.max_vram_gb = 0.0
        with pytest.raises(WFPError) as exc_info:
            ModelPicker().pick(full_spec)
        assert exc_info.value.code == WFPErrorCode.MODEL_NOT_FOUND

    def test_s4_06_manual_repo_used(self, full_spec):
        """S4-06: manual_repo 지정 시 해당 모델 사용."""
        target = "Qwen/Qwen2.5-Coder-7B-Instruct"
        repo = ModelPicker().pick(full_spec, manual_repo=target)
        assert repo == target

    def test_s4_07_unknown_manual_repo_allowed(self, full_spec):
        """S4-07: 카탈로그 없는 manual_repo → 경고 후 허용."""
        unknown = "org/custom-model-xyz"
        repo = ModelPicker().pick(full_spec, manual_repo=unknown)
        assert repo == unknown

    def test_s4_08_spec_updated(self, full_spec):
        """S4-08: pick 후 spec.model_hf_repo 갱신."""
        ModelPicker().pick(full_spec)
        assert full_spec.model_hf_repo != ""


class TestModelDownloader:
    def test_s4_09_cache_reuse(self, tmp_path, full_spec):
        """S4-09: 캐시 존재 시 재다운로드 없이 경로 반환."""
        cache_root = tmp_path / "models"
        downloader = ModelDownloader(cache_root=cache_root)

        # 캐시 시뮬레이션: config.json 생성
        safe_name = full_spec.model_hf_repo.replace("/", "__")
        cached_dir = cache_root / safe_name / "main"
        cached_dir.mkdir(parents=True)
        (cached_dir / "config.json").write_text('{"model_type": "qwen2"}')

        # snapshot_download가 호출되지 않아야 함
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            path = downloader.download(full_spec)
            mock_dl.assert_not_called()

        assert path == str(cached_dir)

    def test_s4_10_insufficient_disk(self, tmp_path, full_spec):
        """S4-10: 디스크 여유 공간 < 필요 공간 → WFP-0404."""
        downloader = ModelDownloader(cache_root=tmp_path / "models")

        import psutil
        mock_usage = MagicMock()
        mock_usage.free = 1 * (1024 ** 3)  # 1GB만 여유

        with patch("psutil.disk_usage", return_value=mock_usage):
            with pytest.raises(WFPError) as exc_info:
                downloader.download(full_spec)
            assert exc_info.value.code == WFPErrorCode.INSUFFICIENT_DISK
