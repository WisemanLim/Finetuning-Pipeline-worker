"""
Stage 5 — 데이터 수집·정제 테스트.

시나리오:
  S5-01  DataCollector — JSONL 파일 업로드 정상 파싱
  S5-02  DataCollector — 존재하지 않는 파일 → WFP-0501
  S5-03  DataCollector — 지원하지 않는 확장자 → WFP-0504
  S5-04  DataCollector — 디렉토리 수집 (txt/md 파일)
  S5-05  DataCollector — 10건 미만 수집 → WFP-0503
  S5-06  DataCleaner — 중복 레코드 제거 확인
  S5-07  DataCleaner — PII(이메일, 전화번호) 마스킹
  S5-08  DataCleaner — 길이 기준 미달 레코드 제거
  S5-09  DataFormatter — instruction 포맷 변환 확인
  S5-10  DataFormatter — train/val/test 분할 비율 확인
  S5-11  DataFormatter — 분할 후 JobSpec 갱신 확인
  S5-12  DataFormatter — 합성 데이터 생성 (시드 기반)
"""

import json
from pathlib import Path

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.types import DataSourceMode
from wfp_worker.stage5_data import DataCleaner, DataCollector, DataFormatter
from wfp_worker.stage5_data.cleaner import CleanConfig


class TestDataCollector:
    def test_s5_01_jsonl_upload(self, tmp_raw_jsonl, minimal_spec, tmp_path):
        """S5-01: JSONL 파일 수집 정상 동작."""
        collector = DataCollector()
        path = collector.collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.UPLOAD,
            source_path=str(tmp_raw_jsonl),
            output_dir=str(tmp_path / "out"),
        )
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) >= 10

    def test_s5_02_missing_file_raises(self, minimal_spec, tmp_path):
        """S5-02: 없는 파일 → WFP-0501."""
        with pytest.raises(WFPError) as exc_info:
            DataCollector().collect(
                spec=minimal_spec,
                source_mode=DataSourceMode.UPLOAD,
                source_path="/nonexistent/file.jsonl",
                output_dir=str(tmp_path / "out"),
            )
        assert exc_info.value.code == WFPErrorCode.DATA_UPLOAD_FAILED

    def test_s5_03_unsupported_extension(self, tmp_path, minimal_spec):
        """S5-03: 지원하지 않는 확장자 → WFP-0504."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("content")
        with pytest.raises(WFPError) as exc_info:
            DataCollector().collect(
                spec=minimal_spec,
                source_mode=DataSourceMode.UPLOAD,
                source_path=str(bad_file),
                output_dir=str(tmp_path / "out"),
            )
        assert exc_info.value.code == WFPErrorCode.DATA_FORMAT_ERROR

    def test_s5_04_directory_collect(self, tmp_path, minimal_spec):
        """S5-04: 디렉토리에서 txt/md 파일 수집."""
        data_dir = tmp_path / "docs"
        data_dir.mkdir()
        for i in range(15):
            (data_dir / f"doc{i}.txt").write_text(
                f"파이썬은 범용 프로그래밍 언어입니다. 예제 {i}번입니다. " * 3
            )
        path = DataCollector().collect(
            spec=minimal_spec,
            source_mode=DataSourceMode.DIRECTORY,
            source_path=str(data_dir),
            output_dir=str(tmp_path / "out"),
        )
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) >= 10

    def test_s5_05_too_few_records(self, tmp_path, minimal_spec):
        """S5-05: 수집 결과 10건 미만 → WFP-0503."""
        small_file = tmp_path / "small.jsonl"
        with small_file.open("w") as f:
            for i in range(3):
                f.write(json.dumps({"text": f"짧은 텍스트 {i}"}) + "\n")

        with pytest.raises(WFPError) as exc_info:
            DataCollector().collect(
                spec=minimal_spec,
                source_mode=DataSourceMode.UPLOAD,
                source_path=str(small_file),
                output_dir=str(tmp_path / "out"),
            )
        assert exc_info.value.code == WFPErrorCode.DATA_TOO_SMALL


class TestDataCleaner:
    def test_s5_06_deduplication(self):
        """S5-06: 중복 레코드 제거."""
        records = [
            {"text": "동일한 텍스트입니다. 파이썬은 범용 프로그래밍 언어로 다양한 분야에 활용됩니다."}
            for _ in range(20)
        ]
        cleaner = DataCleaner(CleanConfig(enable_lang_filter=False))
        result, stats = cleaner.clean(records)
        assert len(result) == 1
        assert stats.removed_dedup == 19

    def test_s5_07_pii_masking(self):
        """S5-07: 이메일·전화번호 마스킹."""
        records = [
            {"text": "연락처: test@example.com 또는 010-1234-5678로 문의하세요."}
        ]
        cleaner = DataCleaner(CleanConfig(enable_dedup=False, enable_lang_filter=False))
        result, _ = cleaner.clean(records)
        assert "test@example.com" not in result[0]["text"]
        assert "010-1234-5678" not in result[0]["text"]
        assert "[EMAIL]" in result[0]["text"]
        assert "[PHONE]" in result[0]["text"]

    def test_s5_08_short_text_removed(self):
        """S5-08: 최소 길이(30자) 미만 텍스트 제거."""
        records = [
            {"text": "짧음"},
            {"text": "충분히 긴 텍스트입니다. 파이썬은 범용 프로그래밍 언어로 매우 유용합니다."},
        ]
        cfg = CleanConfig(min_length=30, enable_dedup=False, enable_lang_filter=False)
        cleaner = DataCleaner(cfg)
        result, stats = cleaner.clean(records)
        assert len(result) == 1
        assert stats.removed_quality == 1


class TestDataFormatter:
    def test_s5_09_instruction_format(self, minimal_spec, tmp_path):
        """S5-09: instruction 포맷 변환 확인."""
        records = [
            {"text": f"파이썬 문법 예제 {i}: for i in range(10): print(i)" * 2}
            for i in range(30)
        ]
        formatter = DataFormatter()
        formatter.format_and_split(minimal_spec, records, output_dir=str(tmp_path / "ds"))

        train_path = tmp_path / "ds" / "train.jsonl"
        first = json.loads(train_path.read_text().splitlines()[0])
        # instruction 또는 messages 필드 존재
        assert "instruction" in first or "messages" in first

    def test_s5_10_split_ratios(self, minimal_spec, tmp_path):
        """S5-10: train/val/test 분할 비율(8:1:1) 확인."""
        records = [{"instruction": f"질문{i}", "input": "", "output": f"답변{i}"} for i in range(100)]
        formatter = DataFormatter()
        formatter.format_and_split(
            minimal_spec, records,
            output_dir=str(tmp_path / "ds"),
            train_ratio=0.8, val_ratio=0.1,
        )
        assert minimal_spec.num_train == 80
        assert minimal_spec.num_val == 10
        assert minimal_spec.num_test == 10

    def test_s5_11_spec_updated(self, minimal_spec, tmp_path):
        """S5-11: 포맷 후 JobSpec 필드 갱신."""
        records = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(20)]
        DataFormatter().format_and_split(
            minimal_spec, records, output_dir=str(tmp_path / "ds")
        )
        assert minimal_spec.dataset_path != ""
        assert minimal_spec.num_train > 0

    def test_s5_12_synthetic_data(self, minimal_spec):
        """S5-12: 합성 데이터 생성 (시드 기반 재현성)."""
        seeds = [{"text": f"파이썬 {i}번 예제입니다."} for i in range(10)]
        formatter = DataFormatter()
        result1 = formatter.generate_synthetic(minimal_spec, seeds, target_count=20)
        result2 = formatter.generate_synthetic(minimal_spec, seeds, target_count=20)
        # 동일 시드 → 동일 결과
        assert len(result1) == len(result2)
        assert result1[0]["instruction"] == result2[0]["instruction"]
