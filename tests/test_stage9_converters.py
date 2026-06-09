"""
Stage 9 보조 — 변환(converters) 테스트.

ML 라이브러리(torch/mlx) 미설치 환경에서도 통과하도록
도구 가용성/디스패치 로직만 검증한다 (실제 변환 실행 제외).
"""

from __future__ import annotations

import pytest

from wfp_worker.common.types import CONVERTIBLE_KINDS, ArtifactKind
from wfp_worker.stage9_package import convert, tooling_status


class TestToolingStatus:
    def test_keys_present(self):
        """모든 변환 포맷 키 + available/hint 구조."""
        st = tooling_status()
        for fmt in ("merged", "gguf", "mlx", "pytorch", "onnx"):
            assert fmt in st
            assert "available" in st[fmt]
            assert "hint" in st[fmt]
            assert isinstance(st[fmt]["available"], bool)

    def test_missing_tool_has_hint(self):
        """미설치 포맷은 안내 힌트 비어있지 않음."""
        st = tooling_status()
        for fmt, info in st.items():
            if not info["available"]:
                assert info["hint"], f"{fmt} 미설치인데 힌트 없음"


class TestConvertibleKinds:
    def test_convertible_excludes_adapter_zip_card(self):
        assert ArtifactKind.ADAPTER not in CONVERTIBLE_KINDS
        assert ArtifactKind.ZIP not in CONVERTIBLE_KINDS
        assert ArtifactKind.MODEL_CARD not in CONVERTIBLE_KINDS

    def test_convertible_includes_new_formats(self):
        for k in (ArtifactKind.MERGED, ArtifactKind.GGUF, ArtifactKind.MLX,
                  ArtifactKind.PYTORCH, ArtifactKind.ONNX):
            assert k in CONVERTIBLE_KINDS


class TestConvertDispatch:
    def test_merged_without_checkpoint_fails(self, minimal_spec, tmp_path):
        """checkpoint 없는 merged 변환은 ConvertFailedError (또는 ToolingMissing).

        torch 미설치면 ToolingMissingError, 설치돼 있으면 checkpoint 부재로
        ConvertFailedError. 둘 중 하나는 반드시 발생.
        """
        minimal_spec.checkpoint_dir = ""
        with pytest.raises(Exception):  # noqa: B017,PT011
            convert(minimal_spec, ArtifactKind.MERGED, tmp_path)
