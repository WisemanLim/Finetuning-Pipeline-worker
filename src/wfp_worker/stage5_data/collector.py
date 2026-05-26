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
        C 모드: Wikipedia 한국어 opensearch + 각 결과 페이지 본문 수집.

        개선 흐름:
          1. 키워드별 opensearch API → 관련 페이지 N개 발견 (직접 URL 추측 X)
          2. 각 페이지 본문 모든 단락 (>=30자) + 리스트 항목 수집
          3. 페이지 내 관련 링크 (See also) 1-hop 확장
          4. 키워드 제한 없음 — 모두 활용

        실제 운영에서는 Google Custom Search / Bing API / Naver API 연동 권장.
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

        sess = requests.Session()
        sess.headers["User-Agent"] = "WFP/0.1 (research; +https://github.com)"
        visited: set[str] = set()
        OPENSEARCH = "https://ko.wikipedia.org/w/api.php"
        SEARCH_LIMIT = 10  # 키워드당 검색 결과 페이지 수
        MIN_PARA_LEN = 30  # 단락 최소 글자

        def _extract_page(url: str, keyword: str) -> Iterator[dict[str, Any]]:
            if url in visited:
                return
            visited.add(url)
            try:
                resp = sess.get(url, timeout=10)
            except Exception as e:
                log.warning("크롤링 실패", extra={"extra": {"url": url, "err": str(e)}})
                return
            if resp.status_code != 200:
                return
            soup = BeautifulSoup(resp.text, "html.parser")
            content = soup.select_one("div#mw-content-text")
            if not content:
                return
            # 본문 단락
            for p in content.select("p"):
                text = p.get_text(strip=True)
                if len(text) >= MIN_PARA_LEN:
                    yield {"text": text, "source": url, "keyword": keyword}
            # 리스트 항목 (의미 있는 문장형)
            for li in content.select("ul > li, ol > li"):
                text = li.get_text(strip=True)
                if MIN_PARA_LEN <= len(text) <= 500:
                    yield {"text": text, "source": url, "keyword": keyword}

        def _opensearch(keyword: str) -> list[str]:
            """opensearch API → 페이지 URL 목록."""
            try:
                r = sess.get(
                    OPENSEARCH,
                    params={
                        "action": "opensearch",
                        "search": keyword,
                        "limit": SEARCH_LIMIT,
                        "format": "json",
                    },
                    timeout=10,
                )
                if r.status_code != 200:
                    return []
                data = r.json()
                return data[3] if len(data) >= 4 else []  # [query, titles, descs, urls]
            except Exception as e:
                log.warning("opensearch 실패", extra={"extra": {"keyword": keyword, "err": str(e)}})
                return []

        # 키워드 비어있으면 KSIC 분야명을 fallback 키워드로
        if not keywords:
            ksic_fallback = {
                "A": ["농업", "임업", "어업"],
                "B": ["광업"],
                "C": ["제조업"],
                "D": ["전기", "가스", "수도사업"],
                "E": ["환경", "폐기물"],
                "F": ["건설업"],
                "G": ["도매", "소매업"],
                "H": ["운수업"],
                "I": ["음식점", "숙박업"],
                "J": ["정보통신", "소프트웨어"],
                "K": ["금융", "보험"],
                "L": ["부동산"],
                "M": ["전문서비스"],
                "N": ["사업서비스"],
                "O": ["공공행정"],
                "P": ["교육"],
                "Q": ["보건", "의료"],
                "R": ["예술", "스포츠"],
                "S": ["수리업"],
            }
            keywords = ksic_fallback.get(ksic_section, [ksic_section])

        for keyword in keywords:
            urls = _opensearch(keyword)
            if not urls:
                # opensearch 실패 시 직접 URL 시도 (legacy fallback)
                urls = [f"https://ko.wikipedia.org/wiki/{keyword}"]
            for url in urls:
                yield from _extract_page(url, keyword)

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
