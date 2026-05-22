"""
Stage 9 — 패키징 테스트.

시나리오:
  S9-01  checkpoint_dir 없으면 WFP-0902
  S9-02  어댑터 파일 존재 시 ADAPTER 아티팩트 생성
  S9-03  model_card.md 항상 생성
  S9-04  ZIP 아카이브 생성 (create_zip=True)
  S9-05  ZIP 파일 내 model_card.md 포함 확인
  S9-06  JobSpec.artifacts 갱신 확인
  S9-07  JobSpec.state = DONE 전이
  S9-08  SHA256 해시 계산 확인 (비어 있지 않음)
  S9-09  ADAPTER 아티팩트 파일 크기 > 0
  S9-10  model_card에 KSIC 섹션 포함 확인
"""

import zipfile
from pathlib import Path

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.types import ArtifactKind, JobState
from wfp_worker.stage9_package import Packager


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """가짜 체크포인트 디렉토리 (adapter 파일 포함)."""
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    # safetensors 파일 생성 (내용은 더미)
    (ckpt / "adapter_model.safetensors").write_bytes(b"dummy_model_data" * 100)
    (ckpt / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (ckpt / "config.json").write_text('{"model_type": "qwen2"}')
    return ckpt


class TestPackager:
    def test_s9_01_no_checkpoint_raises(self, full_spec):
        """S9-01: checkpoint_dir 없으면 WFP-0902."""
        full_spec.checkpoint_dir = ""
        with pytest.raises(WFPError) as exc_info:
            Packager(deploy_root=Path("/tmp/wfp_test")).package(full_spec)
        assert exc_info.value.code == WFPErrorCode.ARTIFACT_UPLOAD_FAILED

    def test_s9_02_adapter_artifact_created(self, full_spec, tmp_path, mock_checkpoint):
        """S9-02: 어댑터 파일 존재 시 ADAPTER 아티팩트."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=False
        )
        kinds = [a.kind for a in artifacts]
        assert ArtifactKind.ADAPTER in kinds

    def test_s9_03_model_card_always_created(self, full_spec, tmp_path, mock_checkpoint):
        """S9-03: model_card.md 항상 생성."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=False
        )
        kinds = [a.kind for a in artifacts]
        assert ArtifactKind.MODEL_CARD in kinds

    def test_s9_04_zip_created(self, full_spec, tmp_path, mock_checkpoint):
        """S9-04: ZIP 아카이브 생성."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=True
        )
        kinds = [a.kind for a in artifacts]
        assert ArtifactKind.ZIP in kinds

    def test_s9_05_zip_contains_model_card(self, full_spec, tmp_path, mock_checkpoint):
        """S9-05: ZIP 내 model_card.md 포함."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=True
        )
        zip_art = next(a for a in artifacts if a.kind == ArtifactKind.ZIP)
        with zipfile.ZipFile(zip_art.uri) as zf:
            names = zf.namelist()
        assert any("model_card.md" in n for n in names)

    def test_s9_06_spec_artifacts_updated(self, full_spec, tmp_path, mock_checkpoint):
        """S9-06: package 후 spec.artifacts 갱신."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        Packager(deploy_root=tmp_path / "deploy").package(full_spec, create_zip=False)
        assert len(full_spec.artifacts) > 0

    def test_s9_07_state_done(self, full_spec, tmp_path, mock_checkpoint):
        """S9-07: package 후 state=DONE."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        Packager(deploy_root=tmp_path / "deploy").package(full_spec, create_zip=False)
        assert full_spec.state == JobState.DONE

    def test_s9_08_sha256_not_empty(self, full_spec, tmp_path, mock_checkpoint):
        """S9-08: ADAPTER 아티팩트 SHA256 비어 있지 않음."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=False
        )
        adapter = next((a for a in artifacts if a.kind == ArtifactKind.ADAPTER), None)
        assert adapter is not None
        assert len(adapter.sha256) == 64  # SHA256 hex = 64자

    def test_s9_09_adapter_size_positive(self, full_spec, tmp_path, mock_checkpoint):
        """S9-09: ADAPTER 아티팩트 파일 크기 > 0."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=False
        )
        adapter = next((a for a in artifacts if a.kind == ArtifactKind.ADAPTER), None)
        assert adapter is not None
        assert adapter.size_bytes > 0

    def test_s9_10_model_card_contains_ksic(self, full_spec, tmp_path, mock_checkpoint):
        """S9-10: model_card에 KSIC 섹션 정보 포함."""
        full_spec.checkpoint_dir = str(mock_checkpoint)
        artifacts = Packager(deploy_root=tmp_path / "deploy").package(
            full_spec, create_zip=False
        )
        card = next(a for a in artifacts if a.kind == ArtifactKind.MODEL_CARD)
        content = Path(card.uri).read_text(encoding="utf-8")
        assert full_spec.ksic_section in content
