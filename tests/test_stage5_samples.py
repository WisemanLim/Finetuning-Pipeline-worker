"""
Stage 5 샘플 파일 통합 테스트.

실제 ops/samples/ 파일을 사용한 end-to-end 검증.
WFP-0503 원인(_extract_text 버그) 수정 검증 포함.

시나리오:
  SS-01  agri_qa.csv 업로드 수집 → 100건 이상
  SS-02  agri_qa.jsonl 업로드 수집 성공
  SS-03  agri_qa.json 업로드 수집 성공
  SS-04  agri_qa.parquet 업로드 수집 (pyarrow 없으면 DATA_FORMAT_ERROR)
  SS-05  _extract_text — instruction+output 합산 (WFP-0503 수정 검증)
  SS-06  _extract_text — 빈 input 필드 무시
  SS-07  agri_qa.csv 품질 필터 통과 (수정 전 전량 제거됐던 케이스)
  SS-08  agri_qa.csv 전체 파이프라인 (collect+clean+format) 성공
  SS-09  documents/ 디렉토리 수집 성공 (txt/md 파일)
  SS-10  documents/ 전체 파이프라인 성공
  SS-11  PDF 파싱 (PyMuPDF 없으면 DATA_FORMAT_ERROR)
  SS-12  DOCX 파싱 (python-docx 없으면 DATA_FORMAT_ERROR)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.types import DataSourceMode
from wfp_worker.stage5_data import DataCleaner, DataCollector, DataFormatter
from wfp_worker.stage5_data.cleaner import CleanConfig

# ─────────────────────────────────────────
# 경로
# ─────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "ops" / "samples" / "upload"
DOCS_DIR = _PROJECT_ROOT / "ops" / "samples" / "documents"

skip_no_upload = pytest.mark.skipif(
    not UPLOAD_DIR.exists(),
    reason=f"샘플 없음: {UPLOAD_DIR}",
)
skip_no_docs = pytest.mark.skipif(
    not DOCS_DIR.exists(),
    reason=f"샘플 없음: {DOCS_DIR}",
)


# ─────────────────────────────────────────
# SS-05 ~ SS-06: _extract_text 단위 테스트
# ─────────────────────────────────────────


class TestExtractTextFix:
    """_extract_text 버그 수정 검증 (WFP-0503 근본 원인)."""

    def test_ss_05_instruction_output_concatenated(self):
        """SS-05: instruction+output 합산 → 품질 필터 통과 길이."""
        rec = {
            "instruction": "벼에 대해 간단히 설명해주세요.",   # 17자 → 단독으론 min_length=30 실패
            "input": "",
            "output": "벼: 쌀 생산을 위한 주요 식량 작물로 한국에서 가장 많이 재배됩니다.",
        }
        result = DataCleaner._extract_text(rec)
        assert "벼에 대해" in result
        assert "주요 식량 작물" in result
        assert len(result) >= 30, f"합산 길이 {len(result)} < 30"

    def test_ss_06_empty_input_ignored(self):
        """SS-06: 빈 input 필드는 합산 제외."""
        rec = {"instruction": "질문입니다.", "input": "", "output": "답변입니다."}
        result = DataCleaner._extract_text(rec)
        # 빈 문자열이 끼어 이중 공백이 생기면 안 됨
        assert "  " not in result
        assert "질문입니다." in result
        assert "답변입니다." in result

    def test_ss_05b_text_field_included(self):
        """text 필드 있으면 합산에 포함."""
        rec = {"text": "본문입니다.", "instruction": "질문"}
        result = DataCleaner._extract_text(rec)
        assert "본문입니다." in result
        assert "질문" in result

    def test_ss_05c_quality_filter_passes_after_fix(self):
        """수정 후: agri_qa 스타일 레코드가 min_length=30 통과."""
        records = [
            {
                "instruction": "벼에 대해 간단히 설명해주세요.",
                "input": "",
                "output": "벼: 쌀 생산을 위한 주요 식량 작물로 한국에서 가장 많이 재배됩니다.",
            },
            {
                "instruction": "배추에 대해 간단히 설명해주세요.",
                "input": "",
                "output": "배추: 김치 원료로 사용되는 십자화과 채소이며 가을철 주요 노지 작물입니다.",
            },
        ]
        cfg = CleanConfig(min_length=30, enable_dedup=False, enable_lang_filter=False)
        result, stats = DataCleaner(cfg).clean(records)
        assert len(result) == 2, f"품질 필터 통과 실패 (제거: {stats.removed_quality})"
        assert stats.removed_quality == 0


# ─────────────────────────────────────────
# SS-01 ~ SS-04: 업로드 파일 수집
# ─────────────────────────────────────────


class TestUploadSamples:
    """ops/samples/upload/ 파일 수집 테스트."""

    @skip_no_upload
    def test_ss_01_csv_collect(self, minimal_spec, tmp_path):
        """SS-01: agri_qa.csv 수집 → 100건 이상."""
        path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.UPLOAD,
            source_path=str(UPLOAD_DIR / "agri_qa.csv"),
            output_dir=str(tmp_path / "out"),
        )
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) >= 100, f"수집 건수 {len(lines)} < 100"

    @skip_no_upload
    def test_ss_02_jsonl_collect(self, minimal_spec, tmp_path):
        """SS-02: agri_qa.jsonl 수집 → 10건 이상."""
        path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.UPLOAD,
            source_path=str(UPLOAD_DIR / "agri_qa.jsonl"),
            output_dir=str(tmp_path / "out"),
        )
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) >= 10

    @skip_no_upload
    def test_ss_03_json_collect(self, minimal_spec, tmp_path):
        """SS-03: agri_qa.json 수집 → 10건 이상."""
        path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.UPLOAD,
            source_path=str(UPLOAD_DIR / "agri_qa.json"),
            output_dir=str(tmp_path / "out"),
        )
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) >= 10

    @skip_no_upload
    def test_ss_04_parquet_collect(self, minimal_spec, tmp_path):
        """SS-04: agri_qa.parquet 수집 (pyarrow 없으면 DATA_FORMAT_ERROR 확인)."""
        try:
            import pyarrow  # noqa: F401
            has_pyarrow = True
        except ImportError:
            has_pyarrow = False

        if not has_pyarrow:
            with pytest.raises(WFPError) as exc:
                DataCollector().collect(
                    spec=minimal_spec,
                    source_mode=DataSourceMode.UPLOAD,
                    source_path=str(UPLOAD_DIR / "agri_qa.parquet"),
                    output_dir=str(tmp_path / "out"),
                )
            assert exc.value.code == WFPErrorCode.DATA_FORMAT_ERROR
        else:
            path = DataCollector().collect(
                spec=minimal_spec,
                source_mode=DataSourceMode.UPLOAD,
                source_path=str(UPLOAD_DIR / "agri_qa.parquet"),
                output_dir=str(tmp_path / "out"),
            )
            lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
            assert len(lines) >= 10


# ─────────────────────────────────────────
# SS-07 ~ SS-08: agri_qa.csv 정제 + 파이프라인
# ─────────────────────────────────────────


class TestCsvCleanPipeline:
    """agri_qa.csv 정제 파이프라인 — WFP-0503 수정 검증."""

    @skip_no_upload
    def test_ss_07_quality_filter_passes(self):
        """SS-07: agri_qa.csv 품질 필터 → 50% 이상 생존 (수정 전엔 0건)."""
        records = []
        with open(UPLOAD_DIR / "agri_qa.csv", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                records.append(dict(row))

        assert len(records) > 0, "CSV 파싱 0건"

        cfg = CleanConfig(
            min_length=30,
            enable_dedup=False,
            enable_lang_filter=False,
        )
        result, stats = DataCleaner(cfg).clean(records)

        assert len(result) > 0, (
            f"품질 필터 후 0건 — WFP-0503 재현. "
            f"removed_quality={stats.removed_quality}/{len(records)}"
        )
        survival_rate = len(result) / len(records)
        assert survival_rate >= 0.5, (
            f"생존율 {survival_rate:.1%} < 50%. "
            f"removed_quality={stats.removed_quality}/{len(records)}"
        )

    @skip_no_upload
    def test_ss_08_csv_full_pipeline(self, minimal_spec, tmp_path):
        """SS-08: agri_qa.csv collect → clean → format 전체 성공."""
        # collect
        raw_path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.UPLOAD,
            source_path=str(UPLOAD_DIR / "agri_qa.csv"),
            output_dir=str(tmp_path / "raw"),
        )
        records = [
            json.loads(line)
            for line in Path(raw_path).read_text().splitlines()
            if line.strip()
        ]
        assert len(records) >= 10

        # clean (lang_filter 비활성 — langdetect 의존성 없이 실행)
        cfg = CleanConfig(enable_lang_filter=False)
        cleaned, stats = DataCleaner(cfg).clean(records)
        assert len(cleaned) > 0, (
            f"clean 후 0건 — WFP-0503. "
            f"removed_quality={stats.removed_quality}/{len(records)}"
        )

        # format
        DataFormatter().format_and_split(
            minimal_spec, cleaned, output_dir=str(tmp_path / "ds")
        )

        train_path = tmp_path / "ds" / "train.jsonl"
        assert train_path.exists(), "train.jsonl 미생성"
        train_records = [
            json.loads(l)
            for l in train_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(train_records) > 0
        first = train_records[0]
        assert "instruction" in first or "messages" in first, (
            f"포맷 변환 실패: {list(first.keys())}"
        )


# ─────────────────────────────────────────
# SS-09 ~ SS-12: 문서 디렉토리
# ─────────────────────────────────────────


class TestDocumentSamples:
    """ops/samples/documents/ 수집 테스트."""

    @skip_no_docs
    def test_ss_09_directory_collect(self, minimal_spec, tmp_path):
        """SS-09: documents/ 디렉토리 수집 → 10건 이상."""
        path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.DIRECTORY,
            source_path=str(DOCS_DIR),
            output_dir=str(tmp_path / "out"),
        )
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        assert len(lines) >= 10, f"수집 건수 {len(lines)} < 10"

    @skip_no_docs
    def test_ss_10_documents_full_pipeline(self, minimal_spec, tmp_path):
        """SS-10: documents/ collect → clean → format 성공."""
        raw_path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.DIRECTORY,
            source_path=str(DOCS_DIR),
            output_dir=str(tmp_path / "raw"),
        )
        records = [
            json.loads(l)
            for l in Path(raw_path).read_text().splitlines()
            if l.strip()
        ]

        cfg = CleanConfig(enable_lang_filter=False)
        cleaned, stats = DataCleaner(cfg).clean(records)
        assert len(cleaned) > 0, f"clean 후 0건. removed_quality={stats.removed_quality}"

        DataFormatter().format_and_split(
            minimal_spec, cleaned, output_dir=str(tmp_path / "ds")
        )
        assert (tmp_path / "ds" / "train.jsonl").exists()

    @skip_no_docs
    def test_ss_11_pdf_parse(self):
        """SS-11: PDF 파싱 (PyMuPDF 없으면 DATA_FORMAT_ERROR)."""
        pdf_file = DOCS_DIR / "11_농업백서_요약.pdf"
        if not pdf_file.exists():
            pytest.skip("PDF 파일 없음")

        try:
            import fitz  # noqa: F401
            has_pymupdf = True
        except ImportError:
            has_pymupdf = False

        if not has_pymupdf:
            with pytest.raises(WFPError) as exc:
                list(DataCollector._parse_pdf(pdf_file))
            assert exc.value.code == WFPErrorCode.DATA_FORMAT_ERROR
        else:
            records = list(DataCollector._parse_pdf(pdf_file))
            assert len(records) > 0
            assert all("text" in r for r in records)

    @skip_no_docs
    def test_ss_12_docx_parse(self):
        """SS-12: DOCX 파싱 (python-docx 없으면 DATA_FORMAT_ERROR)."""
        docx_file = DOCS_DIR / "12_어업자원관리.docx"
        if not docx_file.exists():
            pytest.skip("DOCX 파일 없음")

        try:
            from docx import Document  # noqa: F401
            has_docx = True
        except ImportError:
            has_docx = False

        if not has_docx:
            with pytest.raises(WFPError) as exc:
                list(DataCollector._parse_docx(docx_file))
            assert exc.value.code == WFPErrorCode.DATA_FORMAT_ERROR
        else:
            records = list(DataCollector._parse_docx(docx_file))
            assert len(records) > 0
            assert all("text" in r for r in records)
