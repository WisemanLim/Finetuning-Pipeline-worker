"""
공통 테스트 픽스처.

실제 GPU/HF 다운로드 없이 테스트할 수 있도록
최소한의 mock JobSpec과 HWCaps를 제공한다.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from wfp_worker.common.types import (
    CPUInfo,
    GPUInfo,
    HWCaps,
    JobSpec,
    JobState,
    LoRAConfig,
    Quantization,
    RecommendedPlan,
    TaskType,
    TrainConfig,
    TrainStrategy,
)


# ─────────────────────────────────────────
# 기본 픽스처
# ─────────────────────────────────────────


@pytest.fixture
def minimal_spec() -> JobSpec:
    """최소 필드만 채운 JobSpec (Stage 1 출력 수준)."""
    spec = JobSpec(
        user_id="test-user",
        ksic_section="J",
        ksic_division="62",
        task_type=TaskType.CODE,
        description="소프트웨어 개발 지원 모델",
        language="ko",
    )
    spec.update_stage(stage=1, state=JobState.RUNNING)
    return spec


@pytest.fixture
def hw_caps_nvidia_16gb() -> HWCaps:
    """NVIDIA 16GB VRAM 환경 시뮬레이션."""
    return HWCaps(
        os="linux",
        cpu=CPUInfo(brand="Intel Core i9", cores_physical=8, cores_logical=16, has_avx2=True),
        gpus=[GPUInfo(index=0, name="NVIDIA GeForce RTX 4080", vendor="nvidia", vram_gb=16.0, compute_capability="8.9")],
        ram_gb=32.0,
        disk_free_gb=200.0,
        network_mbps=100.0,
        max_vram_gb=16.0,
        cuda_available=True,
    )


@pytest.fixture
def hw_caps_apple_silicon() -> HWCaps:
    """Apple M-series 환경 시뮬레이션."""
    return HWCaps(
        os="darwin",
        cpu=CPUInfo(brand="Apple M3 Pro", cores_physical=12, cores_logical=12),
        gpus=[GPUInfo(index=0, name="Apple M3 Pro GPU", vendor="apple", vram_gb=36.0)],
        ram_gb=36.0,
        disk_free_gb=300.0,
        network_mbps=500.0,
        max_vram_gb=36.0,
        unified_memory_gb=36.0,
        mps_available=True,
    )


@pytest.fixture
def hw_caps_cpu_only() -> HWCaps:
    """GPU 없는 CPU 전용 환경."""
    return HWCaps(
        os="linux",
        cpu=CPUInfo(brand="Intel Core i5", cores_physical=4, cores_logical=8),
        gpus=[],
        ram_gb=16.0,
        disk_free_gb=50.0,
        network_mbps=50.0,
        max_vram_gb=0.0,
        cuda_available=False,
    )


@pytest.fixture
def recommended_plan_qlora() -> RecommendedPlan:
    """QLoRA 4bit 권장 플랜 (NVIDIA 16GB 기반)."""
    return RecommendedPlan(
        rule_id="nvidia_16gb",
        max_model_params_b=8.0,
        strategies=[TrainStrategy.QLORA],
        quantization=[Quantization.INT4],
        recommended_strategy=TrainStrategy.QLORA,
        recommended_quant=Quantization.INT4,
        backend="cuda",
        max_seq_length=2048,
        batch_size=2,
        gradient_accumulation=8,
    )


@pytest.fixture
def full_spec(minimal_spec: JobSpec, hw_caps_nvidia_16gb: HWCaps, recommended_plan_qlora: RecommendedPlan) -> JobSpec:
    """Stage 6까지 채워진 완전한 JobSpec."""
    spec = minimal_spec
    spec.hw_caps = hw_caps_nvidia_16gb
    spec.recommended_plan = recommended_plan_qlora
    spec.env_ready = True
    spec.python_version = "3.12.6"
    spec.model_hf_repo = "Qwen/Qwen2.5-Coder-7B-Instruct"
    spec.model_local_path = "/tmp/test-model"
    spec.num_train = 100
    spec.num_val = 10
    spec.num_test = 10
    spec.train_config = TrainConfig(
        strategy=TrainStrategy.QLORA,
        quantization=Quantization.INT4,
        max_seq_length=512,
        batch_size=1,
        gradient_accumulation=4,
        num_epochs=1,
        learning_rate=2e-4,
        output_dir="/tmp/test-ckpt",
        lora=LoRAConfig(r=16, lora_alpha=32),
    )
    return spec


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """임시 JSONL 데이터셋 디렉토리 생성."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    records = [
        {"instruction": f"질문 {i}: Python 코드를 작성하세요.", "input": "", "output": f"def func_{i}(): pass"}
        for i in range(30)
    ]

    for split, slice_ in [("train", records[:24]), ("val", records[24:27]), ("test", records[27:])]:
        path = dataset_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for rec in slice_:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return dataset_dir


@pytest.fixture
def tmp_raw_jsonl(tmp_path: Path) -> Path:
    """임시 원시 JSONL 파일 (50건)."""
    raw_path = tmp_path / "raw.jsonl"
    records = [
        {"text": f"파이썬은 인터프리터 언어입니다. 예제 {i}: for i in range(10): print(i)", "source": "test"}
        for i in range(50)
    ]
    with raw_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return raw_path
