"""
Stage 5 — 데이터 정제기 (Cleaner).

PRD §4.5 정제 파이프라인 6단계 구현:
  1. Deduplication (MinHash 근사 중복 제거)
  2. Language Filter (langdetect 또는 규칙 기반 폴백)
  3. Quality Filter (길이, 공백 비율 휴리스틱)
  4. PII 제거 (정규식 기반, Presidio 옵션)
  5. Instruction 포맷 변환은 DataFormatter 담당
  6. 분할은 DataFormatter 담당

각 단계는 독립적으로 on/off 가능한 플래그로 제어.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from wfp_worker.common.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# PII 정규식 패턴 (한국어 + 범용)
# ─────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, str]] = [
    # 이메일
    (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    # 한국 전화번호
    (r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}", "[PHONE]"),
    # 주민등록번호
    (r"\d{6}[-\s]?[1-4]\d{6}", "[SSN]"),
    # 신용카드 번호 (16자리)
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD]"),
    # IP 주소
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
    # URL (완전한 URL만)
    (r"https?://\S+", "[URL]"),
]

_COMPILED_PII = [(re.compile(pat), repl) for pat, repl in _PII_PATTERNS]


@dataclass
class CleanStats:
    """정제 결과 통계."""

    input_count: int = 0
    after_dedup: int = 0
    after_lang_filter: int = 0
    after_quality_filter: int = 0
    after_pii: int = 0
    output_count: int = 0
    removed_dedup: int = 0
    removed_lang: int = 0
    removed_quality: int = 0


@dataclass
class CleanConfig:
    """정제 파이프라인 설정."""

    # 대상 언어 (이 언어만 통과)
    target_language: str = "ko"
    # 최소/최대 텍스트 길이 (문자 수)
    min_length: int = 30
    max_length: int = 10_000
    # 최대 공백 비율
    max_whitespace_ratio: float = 0.3
    # MinHash 근사 중복 제거 해시 수
    dedup_shingle_size: int = 5
    dedup_hash_count: int = 128
    # 기능 on/off
    enable_dedup: bool = True
    enable_lang_filter: bool = True
    enable_quality_filter: bool = True
    enable_pii: bool = True


class DataCleaner:
    """
    Stage 5: 데이터 정제.

    책임:
    - 6단계 정제 파이프라인 순차 실행
    - 각 단계 통계 집계
    - 정제된 레코드 목록 반환
    """

    def __init__(self, config: CleanConfig | None = None) -> None:
        self.config = config or CleanConfig()

    def clean(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], CleanStats]:
        """
        전체 정제 파이프라인 실행.

        Parameters
        ----------
        records:
            DataCollector에서 수집된 원시 레코드 목록.

        Returns
        -------
        (정제된 레코드 목록, 통계)
        """
        stats = CleanStats(input_count=len(records))
        cfg = self.config

        data = records[:]

        # Step 1: 중복 제거
        if cfg.enable_dedup:
            data = self._dedup(data)
            stats.after_dedup = len(data)
            stats.removed_dedup = stats.input_count - stats.after_dedup
        else:
            stats.after_dedup = len(data)

        # Step 2: 언어 필터
        if cfg.enable_lang_filter:
            data = self._lang_filter(data, cfg.target_language)
            stats.after_lang_filter = len(data)
            stats.removed_lang = stats.after_dedup - stats.after_lang_filter
        else:
            stats.after_lang_filter = len(data)

        # Step 3: 품질 필터
        if cfg.enable_quality_filter:
            data = self._quality_filter(data)
            stats.after_quality_filter = len(data)
            stats.removed_quality = stats.after_lang_filter - stats.after_quality_filter
        else:
            stats.after_quality_filter = len(data)

        # Step 4: PII 제거
        if cfg.enable_pii:
            data = [self._remove_pii(rec) for rec in data]
            stats.after_pii = len(data)
        else:
            stats.after_pii = len(data)

        stats.output_count = len(data)

        log.info(
            "Stage 5: 정제 완료",
            extra={
                "stage": 5,
                "extra": {
                    "input": stats.input_count,
                    "output": stats.output_count,
                    "removed_dedup": stats.removed_dedup,
                    "removed_lang": stats.removed_lang,
                    "removed_quality": stats.removed_quality,
                },
            },
        )

        return data, stats

    # ─────────────────────────────────────────
    # Step 1: 중복 제거 (MinHash 근사)
    # ─────────────────────────────────────────

    def _dedup(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        ShingleHash 기반 근사 중복 제거.

        완전한 MinHash LSH 대신 단순 shingle hash 집합으로
        의존성 없이 동작. 정확도보다 속도 우선.
        """
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []

        for rec in records:
            text = self._extract_text(rec)
            sig = self._shingle_hash(text, self.config.dedup_shingle_size)
            if sig not in seen:
                seen.add(sig)
                unique.append(rec)

        return unique

    def _shingle_hash(self, text: str, k: int) -> str:
        """k-shingle 집합의 MD5 해시 (중복 시그니처)."""
        normalized = unicodedata.normalize("NFKC", text.lower())
        tokens = normalized.split()
        if len(tokens) < k:
            # 짧은 텍스트는 전체를 해시
            return hashlib.md5(normalized.encode()).hexdigest()
        shingles = {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}
        combined = "||".join(sorted(shingles))
        return hashlib.md5(combined.encode()).hexdigest()

    # ─────────────────────────────────────────
    # Step 2: 언어 필터
    # ─────────────────────────────────────────

    def _lang_filter(
        self, records: list[dict[str, Any]], target: str
    ) -> list[dict[str, Any]]:
        """target 언어인 레코드만 통과."""
        filtered = []
        for rec in records:
            text = self._extract_text(rec)
            lang = self._detect_language(text)
            if lang == target or lang == "unknown":
                filtered.append(rec)
        return filtered

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        언어 감지.
        langdetect 있으면 사용, 없으면 한글 비율 휴리스틱.
        """
        try:
            from langdetect import detect  # type: ignore[import]
            return detect(text)
        except ImportError:
            pass
        except Exception:
            pass

        # 폴백: 한글 문자 비율로 판단
        if not text:
            return "unknown"
        korean_chars = sum(1 for c in text if "가" <= c <= "힣")
        ratio = korean_chars / max(len(text), 1)
        if ratio > 0.3:
            return "ko"
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        if ascii_chars / max(len(text), 1) > 0.5:
            return "en"
        return "unknown"

    # ─────────────────────────────────────────
    # Step 3: 품질 필터
    # ─────────────────────────────────────────

    def _quality_filter(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """길이, 공백 비율, 반복 패턴 등 품질 기준 필터."""
        cfg = self.config
        filtered = []
        for rec in records:
            text = self._extract_text(rec)

            # 길이 조건
            if not (cfg.min_length <= len(text) <= cfg.max_length):
                continue

            # 공백 비율
            ws_ratio = sum(1 for c in text if c.isspace()) / max(len(text), 1)
            if ws_ratio > cfg.max_whitespace_ratio:
                continue

            # 반복 문자 비율 (스팸성 텍스트 제거)
            if self._is_repetitive(text):
                continue

            filtered.append(rec)
        return filtered

    @staticmethod
    def _is_repetitive(text: str) -> bool:
        """같은 문자가 전체의 50% 이상이면 반복 텍스트로 판단."""
        if not text:
            return False
        most_common_count = max(text.count(c) for c in set(text))
        return most_common_count / len(text) > 0.5

    # ─────────────────────────────────────────
    # Step 4: PII 제거
    # ─────────────────────────────────────────

    @staticmethod
    def _remove_pii(rec: dict[str, Any]) -> dict[str, Any]:
        """레코드 내 텍스트 필드의 PII를 정규식으로 마스킹."""
        result = dict(rec)
        for key in ("text", "instruction", "input", "output", "content"):
            if key in result and isinstance(result[key], str):
                text = result[key]
                for pattern, replacement in _COMPILED_PII:
                    text = pattern.sub(replacement, text)
                result[key] = text
        return result

    # ─────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────

    @staticmethod
    def _extract_text(rec: dict[str, Any]) -> str:
        """레코드에서 텍스트 추출 — 모든 주요 필드 합산.

        instruction+input+output 형식 레코드에서 instruction만 반환하면
        짧은 질문 문장이 품질 필터(min_length)에 걸려 전량 제거됨.
        모든 비어있지 않은 필드를 이어붙여 길이 판정에 사용.
        """
        parts = []
        for key in ("text", "instruction", "input", "output", "content"):
            if key in rec and isinstance(rec[key], str) and rec[key].strip():
                parts.append(rec[key].strip())
        if parts:
            return " ".join(parts)
        return " ".join(str(v) for v in rec.values() if isinstance(v, str))
