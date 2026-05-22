"""
Stage 4 — 모델 선택기.

ModelCatalog에서 후보를 받아 RecommendedPlan의 모델 크기 제약을
추가로 적용한 뒤 최종 모델을 선택한다.

자동 모드: 1순위 후보 자동 채택.
수동 모드: 후보 목록 반환 → 호출자가 hf_repo 지정.
"""

from __future__ import annotations

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import HWCaps, JobSpec, JobState, RecommendedPlan
from .model_catalog import ModelCatalog, ModelEntry

log = get_logger(__name__)


class ModelPicker:
    """
    Stage 4: 후보 모델 선택.

    책임:
    - HW 플랜의 max_model_params_b 기준 필터
    - 언어·카테고리 기준 정렬
    - 자동/수동 선택 처리
    - JobSpec.model_hf_repo 갱신
    """

    def __init__(self, catalog: ModelCatalog | None = None) -> None:
        self._catalog = catalog or ModelCatalog()

    def pick(
        self,
        spec: JobSpec,
        manual_repo: str | None = None,
    ) -> str:
        """
        최종 모델 HF repo id 반환 및 JobSpec 갱신.

        Parameters
        ----------
        spec:
            Stage 3까지 채워진 JobSpec.
        manual_repo:
            None이면 자동 선택, 값이 있으면 해당 repo 사용.

        Returns
        -------
        str
            선택된 HuggingFace repo id.

        Raises
        ------
        WFPError(WFP-0402)
            후보가 없거나 manual_repo가 카탈로그에 없을 때.
        """
        plan: RecommendedPlan | None = spec.recommended_plan
        hw: HWCaps | None = spec.hw_caps
        max_vram = hw.max_vram_gb if hw else 0.0
        max_params = plan.max_model_params_b if plan else 70.0

        log.info(
            "Stage 4: 모델 선택 시작",
            extra={
                "stage": 4,
                "job_id": str(spec.job_id),
                "extra": {
                    "ksic_section": spec.ksic_section,
                    "max_vram_gb": max_vram,
                    "max_params_b": max_params,
                    "manual_repo": manual_repo,
                },
            },
        )

        candidates = self._catalog.get_candidates(
            ksic_section=spec.ksic_section,
            max_vram_gb=max_vram,
            language=spec.language,
        )

        # max_model_params_b 제약 추가
        candidates = [m for m in candidates if m.params_b <= max_params]

        if not candidates:
            raise WFPError(
                code=WFPErrorCode.MODEL_NOT_FOUND,
                message="HW 제약에 맞는 후보 모델이 없습니다.",
                stage=4,
                detail=f"max_vram={max_vram}GB, max_params={max_params}B",
            )

        # 수동 선택 검증
        if manual_repo:
            matched = next((m for m in candidates if m.hf_repo == manual_repo), None)
            if not matched:
                # 카탈로그 외 repo는 경고 후 허용 (사용자 책임)
                log.warning(
                    "지정된 repo가 카탈로그에 없음 — 그대로 사용",
                    extra={"stage": 4, "extra": {"manual_repo": manual_repo}},
                )
                selected_repo = manual_repo
            else:
                selected_repo = matched.hf_repo
        else:
            # 자동 선택 — 1순위
            selected_repo = candidates[0].hf_repo

        spec.model_hf_repo = selected_repo
        spec.update_stage(stage=4, state=JobState.RUNNING)

        log.info(
            "Stage 4: 모델 선택 완료",
            extra={
                "stage": 4,
                "job_id": str(spec.job_id),
                "extra": {"selected": selected_repo, "candidates": [m.hf_repo for m in candidates]},
            },
        )

        return selected_repo

    def list_candidates(self, spec: JobSpec) -> list[ModelEntry]:
        """UI 표시용 후보 목록 반환 (pick 없이)."""
        plan = spec.recommended_plan
        hw = spec.hw_caps
        max_vram = hw.max_vram_gb if hw else 0.0
        max_params = plan.max_model_params_b if plan else 70.0

        candidates = self._catalog.get_candidates(
            ksic_section=spec.ksic_section,
            max_vram_gb=max_vram,
            language=spec.language,
        )
        return [m for m in candidates if m.params_b <= max_params]
