"""
Stage 5 — 데이터 수집기 (Collector).

PRD §4.5 데이터 소스 모드 A/B/C/D를 구현.

A. UPLOAD  : JSONL/CSV/Parquet 파일 로드
B. DIR     : 로컬 디렉토리에서 PDF/DOCX/HTML/MD/TXT 파싱
C. WEB     : 키워드 검색 기반 크롤링 (화이트리스트 도메인)
D. HYBRID  : A/B + C 결합

실제 웹 크롤링은 `requests` + `BeautifulSoup` 기반으로 구현.
PDF/DOCX 파싱은 선택 의존성(PyMuPDF, python-docx).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import DataSourceMode, JobSpec, JobState

log = get_logger(__name__)

# 크롤링 허용 도메인 화이트리스트 (운영 환경에서는 DB로 관리)
_WHITELIST_DOMAINS: frozenset[str] = frozenset([
    "namu.wiki", "ko.wikipedia.org", "en.wikipedia.org",
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
    "aihub.or.kr", "data.go.kr", "kostat.go.kr",
    "github.com", "docs.python.org",
])

# 지원 파일 확장자 → 파서 함수 이름
_EXT_MAP: dict[str, str] = {
    ".jsonl": "_parse_jsonl",
    ".json":  "_parse_json",   # JSON array or single object
    ".csv":   "_parse_csv",
    ".parquet": "_parse_parquet",
    ".txt":   "_parse_text",
    ".md":    "_parse_text",
    ".html":  "_parse_html",
    ".htm":   "_parse_html",
    ".pdf":   "_parse_pdf",
    ".docx":  "_parse_docx",
}


class DataCollector:
    """
    Stage 5: 데이터 수집.

    책임:
    - 소스 모드에 따른 원시 텍스트 수집
    - 최소 레코드 수 검증
    - 수집 결과를 JSONL 임시 파일에 저장
    - JobSpec.dataset_path 갱신 (정제 전 원시 경로)
    """

    MIN_RECORDS = 10  # 최소 수집 레코드 수

    def collect(
        self,
        spec: JobSpec,
        source_mode: DataSourceMode,
        source_path: str = "",
        keywords: list[str] | None = None,
        output_dir: str = "",
    ) -> str:
        """
        데이터 수집 실행.

        Parameters
        ----------
        spec:
            Stage 4까지 채워진 JobSpec.
        source_mode:
            수집 모드 (UPLOAD/DIR/WEB/SYNTHETIC).
        source_path:
            UPLOAD 또는 DIR 모드 시 파일/디렉토리 경로.
        keywords:
            WEB 모드 검색 키워드 목록.
        output_dir:
            수집 결과 저장 디렉토리. 미지정 시 ~/.wfp/jobs/<job_id>/raw.

        Returns
        -------
        str
            수집된 원시 JSONL 파일 경로.
        """
        spec.data_source_mode = source_mode

        out_dir = Path(output_dir) if output_dir else (
            Path.home() / ".wfp" / "jobs" / str(spec.job_id) / "raw"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "collected.jsonl"

        log.info(
            "Stage 5: 데이터 수집 시작",
            extra={"stage": 5, "job_id": str(spec.job_id), "extra": {"mode": source_mode.value}},
        )

        records: list[dict[str, Any]] = []

        if source_mode == DataSourceMode.UPLOAD:
            records = list(self._collect_upload(source_path))

        elif source_mode == DataSourceMode.DIRECTORY:
            records = list(self._collect_directory(source_path))

        elif source_mode == DataSourceMode.WEB:
            records = list(self._collect_web(keywords or [], spec.ksic_section))

        elif source_mode == DataSourceMode.SYNTHETIC:
            # 합성 데이터는 DataFormatter에서 생성 — 여기서는 빈 시드 반환
            log.info("합성 모드 — 수집 생략, formatter에서 생성", extra={"stage": 5})
            records = []

        # 최소 레코드 수 검증 (합성 모드 제외)
        if source_mode != DataSourceMode.SYNTHETIC and len(records) < self.MIN_RECORDS:
            raise WFPError(
                code=WFPErrorCode.DATA_TOO_SMALL,
                message=f"수집 데이터 부족. 최소 {self.MIN_RECORDS}건 필요, 수집: {len(records)}건",
                stage=5,
            )

        # JSONL 저장
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        spec.update_stage(stage=5, state=JobState.RUNNING)

        log.info(
            "Stage 5: 수집 완료",
            extra={"stage": 5, "job_id": str(spec.job_id),
                   "extra": {"records": len(records), "path": str(out_path)}},
        )

        return str(out_path)

    # ─────────────────────────────────────────
    # 모드별 수집 구현
    # ─────────────────────────────────────────

    def _collect_upload(self, source_path: str) -> Iterator[dict[str, Any]]:
        """A 모드: 업로드된 파일 파싱."""
        path = Path(source_path)
        if not path.exists():
            raise WFPError(
                code=WFPErrorCode.DATA_UPLOAD_FAILED,
                message=f"파일 없음: {source_path}",
                stage=5,
            )

        ext = path.suffix.lower()
        parser_name = _EXT_MAP.get(ext)
        if not parser_name:
            raise WFPError(
                code=WFPErrorCode.DATA_FORMAT_ERROR,
                message=f"지원하지 않는 파일 형식: {ext}",
                stage=5,
                detail=f"지원 형식: {list(_EXT_MAP.keys())}",
            )

        parser = getattr(self, parser_name)
        yield from parser(path)

    def _collect_directory(self, dir_path: str) -> Iterator[dict[str, Any]]:
        """B 모드: 디렉토리 재귀 스캔."""
        base = Path(dir_path)
        if not base.exists() or not base.is_dir():
            raise WFPError(
                code=WFPErrorCode.DATA_DIR_NOT_FOUND,
                message=f"디렉토리 없음: {dir_path}",
                stage=5,
            )

        for file in sorted(base.rglob("*")):
            if not file.is_file():
                continue
            ext = file.suffix.lower()
            parser_name = _EXT_MAP.get(ext)
            if parser_name:
                try:
                    yield from getattr(self, parser_name)(file)
                except Exception as e:
                    log.warning(
                        "파일 파싱 실패 — 건너뜀",
                        extra={"extra": {"file": str(file), "err": str(e)}},
                    )

    def _collect_web(
        self, keywords: list[str], ksic_section: str
    ) -> Iterator[dict[str, Any]]:
        """
        C 모드: 화이트리스트 기반 간단 크롤링.

        실제 운영에서는 검색 API(Google Custom Search, Bing) 연동 권장.
        현재는 Wikipedia 한국어 검색만 구현 (데모 수준).
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.DATA_CLEAN_FAILED,
                message="웹 수집에 requests, beautifulsoup4 필요. pip install requests beautifulsoup4",
                stage=5,
            ) from None

        for keyword in keywords[:5]:  # 키워드당 Wikipedia 검색
            url = f"https://ko.wikipedia.org/wiki/{keyword}"
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "WFP/0.1"})
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                # 본문 단락 추출
                for p in soup.select("div#mw-content-text p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        yield {"text": text, "source": url, "keyword": keyword}
            except Exception as e:
                log.warning("크롤링 실패", extra={"extra": {"url": url, "err": str(e)}})

    # ─────────────────────────────────────────
    # 파일 파서
    # ─────────────────────────────────────────

    @staticmethod
    def _parse_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    @staticmethod
    def _parse_json(path: Path) -> Iterator[dict[str, Any]]:
        """JSON 파일 파싱 — 배열([{...}]) 또는 단일 객체({...}) 모두 지원."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data

    @staticmethod
    def _parse_csv(path: Path) -> Iterator[dict[str, Any]]:
        import csv

        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)

    @staticmethod
    def _parse_parquet(path: Path) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore[import]

            table = pq.read_table(str(path))
            for batch in table.to_batches():
                for row in batch.to_pydict().items():
                    pass
            # 간단 변환
            records = table.to_pydict()
            keys = list(records.keys())
            n = len(records[keys[0]]) if keys else 0
            for i in range(n):
                yield {k: records[k][i] for k in keys}
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.DATA_FORMAT_ERROR,
                message="Parquet 파싱에 pyarrow 필요. pip install pyarrow",
                stage=5,
            ) from None

    @staticmethod
    def _parse_text(path: Path) -> Iterator[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # 단락 분리 (빈 줄 기준)
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:
                yield {"text": para, "source": path.name}

    @staticmethod
    def _parse_html(path: Path) -> Iterator[dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # BeautifulSoup 없으면 텍스트 추출만
            text = re.sub(r"<[^>]+>", "", path.read_text(errors="ignore"))
            yield {"text": text.strip(), "source": path.name}
            return

        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        for para in text.split("\n"):
            if len(para.strip()) > 30:
                yield {"text": para.strip(), "source": path.name}

    @staticmethod
    def _parse_pdf(path: Path) -> Iterator[dict[str, Any]]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.DATA_FORMAT_ERROR,
                message="PDF 파싱에 PyMuPDF 필요. pip install pymupdf",
                stage=5,
            ) from None

        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc):
            text = page.get_text()
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 30:
                    yield {"text": para, "source": path.name, "page": page_num + 1}

    @staticmethod
    def _parse_docx(path: Path) -> Iterator[dict[str, Any]]:
        try:
            from docx import Document  # python-docx
        except ImportError:
            raise WFPError(
                code=WFPErrorCode.DATA_FORMAT_ERROR,
                message="DOCX 파싱에 python-docx 필요. pip install python-docx",
                stage=5,
            ) from None

        doc = Document(str(path))
        for para in doc.paragraphs:
            if len(para.text.strip()) > 30:
                yield {"text": para.text.strip(), "source": path.name}
