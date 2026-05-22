"""
Stage 5 — 데이터 포매터 (Formatter).

PRD §4.5 정제 파이프라인 Step 5~6 구현:
  5. Instruction 포맷 변환
     {"instruction":..., "input":..., "output":...} 또는 chat-messages 형식
  6. Train/Val/Test 분할 (기본 8:1:1)

합성 데이터(Self-Instruct) 생성 옵션도 제공.
생성된 데이터셋은 HuggingFace datasets 포맷으로 저장.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from wfp_worker.common.errors import WFPError, WFPErrorCode
from wfp_worker.common.logger import get_logger
from wfp_worker.common.types import JobSpec, JobState, TaskType

log = get_logger(__name__)

# 태스크 타입별 instruction 포맷 템플릿
_TASK_TEMPLATES: dict[TaskType, dict[str, str]] = {
    TaskType.CHAT: {
        "system": "당신은 {domain} 분야의 전문 AI 어시스턴트입니다.",
        "instruction_prefix": "",
    },
    TaskType.QA: {
        "system": "아래 질문에 정확하고 간결하게 답변하세요.",
        "instruction_prefix": "질문: ",
    },
    TaskType.SUMMARY: {
        "system": "아래 텍스트를 간결하게 요약하세요.",
        "instruction_prefix": "텍스트: ",
    },
    TaskType.CLASSIFICATION: {
        "system": "아래 텍스트를 주어진 카테고리 중 하나로 분류하세요.",
        "instruction_prefix": "",
    },
    TaskType.CODE: {
        "system": "당신은 전문 소프트웨어 개발자입니다. 요청에 따라 코드를 작성하세요.",
        "instruction_prefix": "",
    },
}


class DataFormatter:
    """
    Stage 5: 데이터 포맷 변환 및 분할.

    책임:
    - 원시 레코드 → Instruction format 변환
    - chat-messages 형식(OpenAI 호환) 지원
    - Train/Val/Test 분할
    - HuggingFace DatasetDict 또는 JSONL 파일 저장
    - JobSpec.num_train/val/test 갱신
    """

    def format_and_split(
        self,
        spec: JobSpec,
        records: list[dict[str, Any]],
        output_dir: str = "",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        use_chat_format: bool = False,
    ) -> str:
        """
        포맷 변환 후 분할하여 저장.

        Parameters
        ----------
        spec:
            TaskType, JobId 등이 채워진 JobSpec.
        records:
            DataCleaner를 통과한 정제 레코드 목록.
        output_dir:
            저장 디렉토리. 미지정 시 ~/.wfp/jobs/<job_id>/dataset.
        train_ratio / val_ratio:
            분할 비율. test = 1 - train - val.
        seed:
            셔플 랜덤 시드.
        use_chat_format:
            True면 chat-messages 형식, False면 instruction 형식.

        Returns
        -------
        str
            데이터셋이 저장된 디렉토리 경로.
        """
        if not records:
            raise WFPError(
                code=WFPErrorCode.DATA_TOO_SMALL,
                message="포맷 변환할 레코드가 없습니다.",
                stage=5,
            )

        out_dir = Path(output_dir) if output_dir else (
            Path.home() / ".wfp" / "jobs" / str(spec.job_id) / "dataset"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # 포맷 변환
        if use_chat_format:
            formatted = [self._to_chat_format(rec, spec) for rec in records]
        else:
            formatted = [self._to_instruction_format(rec, spec) for rec in records]

        # 셔플 및 분할
        rng = random.Random(seed)
        rng.shuffle(formatted)

        n = len(formatted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": formatted[:n_train],
            "val": formatted[n_train:n_train + n_val],
            "test": formatted[n_train + n_val:],
        }

        # JSONL 저장
        for split_name, split_data in splits.items():
            split_path = out_dir / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # JobSpec 갱신
        spec.dataset_path = str(out_dir)
        spec.num_train = len(splits["train"])
        spec.num_val = len(splits["val"])
        spec.num_test = len(splits["test"])
        spec.update_stage(stage=5, state=JobState.RUNNING)

        log.info(
            "Stage 5: 포맷 변환 및 분할 완료",
            extra={
                "stage": 5,
                "job_id": str(spec.job_id),
                "extra": {
                    "total": n,
                    "train": spec.num_train,
                    "val": spec.num_val,
                    "test": spec.num_test,
                    "path": str(out_dir),
                },
            },
        )

        return str(out_dir)

    # ─────────────────────────────────────────
    # 포맷 변환기
    # ─────────────────────────────────────────

    @staticmethod
    def _to_instruction_format(
        rec: dict[str, Any], spec: JobSpec
    ) -> dict[str, Any]:
        """
        원시 레코드 → Alpaca instruction 형식 변환.

        {"instruction": ..., "input": ..., "output": ...}
        """
        # 이미 instruction 형식이면 그대로 사용
        if "instruction" in rec and "output" in rec:
            return {
                "instruction": str(rec.get("instruction", "")),
                "input": str(rec.get("input", "")),
                "output": str(rec.get("output", "")),
            }

        # 단순 텍스트 → QA 스타일로 래핑
        text = str(rec.get("text", rec.get("content", "")))
        template = _TASK_TEMPLATES.get(spec.task_type, _TASK_TEMPLATES[TaskType.QA])

        return {
            "instruction": template["instruction_prefix"] + text,
            "input": "",
            "output": "",  # 실제 학습 데이터는 output이 있어야 함
        }

    @staticmethod
    def _to_chat_format(
        rec: dict[str, Any], spec: JobSpec
    ) -> dict[str, Any]:
        """
        원시 레코드 → OpenAI chat-messages 형식 변환.

        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
        """
        template = _TASK_TEMPLATES.get(spec.task_type, _TASK_TEMPLATES[TaskType.CHAT])

        # 도메인 이름 추론 (KSIC 섹션 기반)
        domain_map = {
            "J": "정보통신", "K": "금융", "Q": "보건의료",
            "C": "제조", "G": "유통", "M": "전문기술",
        }
        domain = domain_map.get(spec.ksic_section, "전문")
        system_msg = template["system"].format(domain=domain)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

        # 이미 messages 형식이면 병합
        if "messages" in rec and isinstance(rec["messages"], list):
            messages.extend(rec["messages"])
        elif "instruction" in rec:
            user_content = str(rec["instruction"])
            if rec.get("input"):
                user_content += f"\n\n{rec['input']}"
            messages.append({"role": "user", "content": user_content})
            if rec.get("output"):
                messages.append({"role": "assistant", "content": str(rec["output"])})
        else:
            text = str(rec.get("text", rec.get("content", "")))
            messages.append({"role": "user", "content": text})

        return {"messages": messages}

    def generate_synthetic(
        self,
        spec: JobSpec,
        seed_records: list[dict[str, Any]],
        target_count: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Self-Instruct 스타일 합성 데이터 생성 (Teacher 모델 없는 경우 규칙 기반).

        실제 운영에서는 GPT-4o/Claude API를 teacher로 사용.
        현재는 시드 데이터 재조합 방식으로 데모 구현.
        """
        if not seed_records:
            log.warning("시드 데이터 없음 — 합성 생략")
            return []

        synthetic: list[dict[str, Any]] = []
        rng = random.Random(42)

        for _ in range(min(target_count, len(seed_records) * 3)):
            base = rng.choice(seed_records)
            text = str(base.get("text", base.get("instruction", "")))

            # 간단한 질문 생성 패턴
            q_templates = [
                f"{text[:50]}에 대해 설명하세요.",
                f"{text[:30]}의 주요 특징은 무엇입니까?",
                f"다음에 대한 요약을 작성하세요: {text[:50]}",
            ]
            instruction = rng.choice(q_templates)

            synthetic.append({
                "instruction": instruction,
                "input": "",
                "output": text,
                "synthetic": True,
            })

        log.info(
            "합성 데이터 생성 완료",
            extra={"stage": 5, "extra": {"count": len(synthetic)}},
        )
        return synthetic
