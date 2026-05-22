"""
Stage 1 — 목표 정의 테스트.

시나리오:
  S1-01  정상 입력 → JobSpec 생성 검증
  S1-02  모든 유효 KSIC 섹션(A~U) 수용
  S1-03  잘못된 KSIC 섹션 → WFP-0101
  S1-04  잘못된 태스크 타입 → WFP-0102
  S1-05  설명 10자 미만 → WFP-0103
  S1-06  언어 자동 추론 (미입력 시 'ko' 기본)
  S1-07  대소문자 무관 섹션 입력
  S1-08  KSIC 중분류 유효값/무효값
  S1-09  JobSpec UUID 고유성 확인
  S1-10  JobSpec 초기 상태 RUNNING 확인
"""

import pytest

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.types import JobState, TaskType
from wfp_worker.stage1_init import GoalBuilder


@pytest.fixture
def builder() -> GoalBuilder:
    return GoalBuilder()


class TestGoalBuilderHappyPath:
    def test_s1_01_basic_success(self, builder: GoalBuilder):
        """S1-01: 정상 입력으로 JobSpec 생성."""
        spec = builder.build(
            ksic_section="J",
            task_type="code",
            description="정보통신 분야 소프트웨어 개발 지원 모델",
        )
        assert spec.ksic_section == "J"
        assert spec.task_type == TaskType.CODE
        assert len(str(spec.job_id)) == 36  # UUID4 형식

    def test_s1_02_all_valid_sections(self, builder: GoalBuilder):
        """S1-02: A~U 모든 KSIC 대분류 수용."""
        for section in "ABCDEFGHIJKLMNOPQRSTU":
            spec = builder.build(
                ksic_section=section,
                task_type="chat",
                description="테스트용 모델입니다 - 충분한 설명",
            )
            assert spec.ksic_section == section

    def test_s1_06_language_auto_infer(self, builder: GoalBuilder):
        """S1-06: 언어 미입력 시 'ko' 기본값."""
        spec = builder.build(
            ksic_section="K",
            task_type="qa",
            description="금융 데이터 분석 모델입니다.",
        )
        assert spec.language == "ko"

    def test_s1_07_case_insensitive_section(self, builder: GoalBuilder):
        """S1-07: 소문자 섹션 입력도 정규화."""
        spec = builder.build(
            ksic_section="j",
            task_type="code",
            description="소프트웨어 개발 지원 모델",
        )
        assert spec.ksic_section == "J"

    def test_s1_08_valid_division(self, builder: GoalBuilder):
        """S1-08: 유효한 KSIC 중분류."""
        spec = builder.build(
            ksic_section="J",
            ksic_division="62",
            task_type="code",
            description="소프트웨어 개발 지원 모델",
        )
        assert spec.ksic_division == "62"

    def test_s1_09_unique_job_ids(self, builder: GoalBuilder):
        """S1-09: 반복 호출 시 UUID 고유성."""
        specs = [
            builder.build("J", "chat", "모델 설명입니다. 충분히 긴 설명.")
            for _ in range(5)
        ]
        ids = [str(s.job_id) for s in specs]
        assert len(set(ids)) == 5

    def test_s1_10_initial_state_running(self, builder: GoalBuilder):
        """S1-10: 생성된 JobSpec 초기 상태 RUNNING."""
        spec = builder.build("J", "chat", "모델 설명입니다. 충분히 긴 설명.")
        assert spec.state == JobState.RUNNING
        assert spec.current_stage == 1


class TestGoalBuilderErrors:
    def test_s1_03_invalid_section(self, builder: GoalBuilder):
        """S1-03: 잘못된 KSIC 섹션 → WFP-0101."""
        with pytest.raises(WFPError) as exc_info:
            builder.build("Z", "chat", "테스트 설명입니다.")
        assert exc_info.value.code == WFPErrorCode.INVALID_KSIC_SECTION

    def test_s1_04_invalid_task_type(self, builder: GoalBuilder):
        """S1-04: 잘못된 태스크 타입 → WFP-0102."""
        with pytest.raises(WFPError) as exc_info:
            builder.build("J", "invalid_task", "테스트 설명입니다.")
        assert exc_info.value.code == WFPErrorCode.INVALID_TASK_TYPE

    def test_s1_05_short_description(self, builder: GoalBuilder):
        """S1-05: 설명 10자 미만 → WFP-0103."""
        with pytest.raises(WFPError) as exc_info:
            builder.build("J", "chat", "짧음")
        assert exc_info.value.code == WFPErrorCode.MISSING_DESCRIPTION

    def test_s1_08_invalid_division(self, builder: GoalBuilder):
        """S1-08: 잘못된 KSIC 중분류 → WFP-0101."""
        with pytest.raises(WFPError) as exc_info:
            builder.build("J", "chat", "충분한 설명입니다.", ksic_division="999")
        assert exc_info.value.code == WFPErrorCode.INVALID_KSIC_SECTION
