"""
Stage 1 — 목표 정의 (Init/Goal).

사용자 입력(카테고리, 태스크 타입, 설명)을 받아
파이프라인 전체에서 참조하는 JobSpec을 생성·검증한다.

유효한 KSIC 대분류(A~U)와 태스크 타입만 허용.
"""

from __future__ import annotations

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobSpec, JobState, TaskType

log = get_logger(__name__)

# KSIC 대분류 코드 (A~U, 21개 섹션)
VALID_KSIC_SECTIONS: frozenset[str] = frozenset(
    "A B C D E F G H I J K L M N O P Q R S T U".split()
)

# KSIC 중분류 코드 범위 (01~99, 두 자리 문자열)
VALID_KSIC_DIVISIONS: frozenset[str] = frozenset(f"{i:02d}" for i in range(1, 100))

# 언어 화이트리스트
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(["ko", "en", "ja", "zh"])

# 카테고리별 기본 언어 추론 (향후 DB category_preset으로 대체)
_DEFAULT_LANGUAGE_BY_SECTION: dict[str, str] = {
    "A": "ko", "B": "ko", "C": "ko", "D": "ko", "E": "ko",
    "F": "ko", "G": "ko", "H": "ko", "I": "ko", "J": "ko",
    "K": "ko", "L": "ko", "M": "ko", "N": "ko", "O": "ko",
    "P": "ko", "Q": "ko", "R": "ko", "S": "ko", "T": "ko", "U": "ko",
}


class GoalBuilder:
    """
    Stage 1: 사용자 목표를 JobSpec으로 변환.

    책임:
    - 입력값 검증 (섹션, 중분류, 태스크 타입)
    - JobSpec 초기화 및 state=RUNNING 전이
    - 검증 실패 시 WFPError(stage=1) 발생
    """

    def build(
        self,
        ksic_section: str,
        task_type: str | TaskType,
        description: str,
        ksic_division: str = "",
        language: str = "",
        user_id: str = "",
    ) -> JobSpec:
        """
        검증된 JobSpec 생성.

        Parameters
        ----------
        ksic_section:
            KSIC 대분류 코드 (A~U). 대소문자 무관.
        task_type:
            파인튜닝 목적 태스크 타입 (chat/qa/summary/classification/code).
        description:
            사용자가 입력한 모델 용도 설명 (최소 10자).
        ksic_division:
            KSIC 중분류 코드 (01~99). 선택 항목.
        language:
            학습 언어 코드. 미입력 시 섹션 기반 추론.
        user_id:
            요청 사용자 ID (DB 연동 시 사용).
        """
        log.info("Stage 1 시작", extra={"stage": 1, "user_id": user_id})

        # 섹션 정규화 및 검증
        section = ksic_section.strip().upper()
        if section not in VALID_KSIC_SECTIONS:
            raise WFPError(
                code=WFPErrorCode.INVALID_KSIC_SECTION,
                message=f"유효하지 않은 KSIC 섹션: '{section}'. 허용값: A~U",
                stage=1,
            )

        # 중분류 검증 (입력 시만)
        division = ksic_division.strip().zfill(2) if ksic_division.strip() else ""
        if division and division not in VALID_KSIC_DIVISIONS:
            raise WFPError(
                code=WFPErrorCode.INVALID_KSIC_SECTION,
                message=f"유효하지 않은 KSIC 중분류: '{division}'. 허용값: 01~99",
                stage=1,
                detail=f"입력값: {ksic_division!r}",
            )

        # 태스크 타입 검증
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type.lower())
            except ValueError:
                valid = [t.value for t in TaskType]
                raise WFPError(
                    code=WFPErrorCode.INVALID_TASK_TYPE,
                    message=f"유효하지 않은 태스크 타입: '{task_type}'. 허용값: {valid}",
                    stage=1,
                ) from None

        # 설명 최소 길이 검사
        description = description.strip()
        if len(description) < 10:
            raise WFPError(
                code=WFPErrorCode.MISSING_DESCRIPTION,
                message="설명(description)은 최소 10자 이상 입력해야 합니다.",
                stage=1,
                detail=f"현재 길이: {len(description)}자",
            )

        # 언어 결정
        lang = language.strip().lower() if language.strip() else ""
        if not lang:
            lang = _DEFAULT_LANGUAGE_BY_SECTION.get(section, "ko")
        if lang not in SUPPORTED_LANGUAGES:
            lang = "ko"  # 미지원 언어는 한국어로 폴백

        # JobSpec 생성
        spec = JobSpec(
            user_id=user_id,
            ksic_section=section,
            ksic_division=division,
            task_type=task_type,
            description=description,
            language=lang,
        )
        spec.update_stage(stage=1, state=JobState.RUNNING)

        log.info(
            "Stage 1 완료 — JobSpec 생성",
            extra={
                "job_id": str(spec.job_id),
                "stage": 1,
                "extra": {
                    "ksic_section": section,
                    "task_type": task_type.value,
                    "language": lang,
                },
            },
        )

        return spec
