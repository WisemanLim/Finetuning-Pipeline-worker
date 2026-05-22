"""
공통 데이터 타입 정의.

파이프라인 전 단계에서 공유되는 Pydantic 모델 및 Enum.
JobSpec이 모든 스테이지의 입출력 컨테이너 역할을 한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


def _utc_now() -> datetime:
    """timezone-aware UTC 현재 시각 (Python 3.12 deprecation 회피)."""
    return datetime.now(timezone.utc)

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# Enum 정의
# ─────────────────────────────────────────


class TaskType(str, Enum):
    """파인튜닝 목적 태스크 타입."""

    CHAT = "chat"
    QA = "qa"
    SUMMARY = "summary"
    CLASSIFICATION = "classification"
    CODE = "code"


class TrainStrategy(str, Enum):
    """학습 전략."""

    FULL_FT = "full_ft"   # Full Fine-tuning (모든 파라미터)
    LORA = "lora"         # Low-Rank Adaptation
    QLORA = "qlora"       # Quantized LoRA (4bit)
    DPO = "dpo"           # Direct Preference Optimization
    ORPO = "orpo"         # Odds Ratio Preference Optimization


class Quantization(str, Enum):
    """모델 양자화 수준."""

    NONE = "none"    # bf16 / fp16
    INT8 = "int8"    # 8-bit 양자화
    INT4 = "int4"    # 4-bit 양자화 (QLoRA 기본)


class DataSourceMode(str, Enum):
    """데이터 소스 유형 (Stage 5)."""

    UPLOAD = "upload"    # 사용자 파일 업로드
    DIRECTORY = "dir"    # 로컬 디렉토리
    WEB = "web"          # 인터넷 검색 수집
    SYNTHETIC = "synth"  # 합성 데이터 생성


class JobState(str, Enum):
    """잡 라이프사이클 상태."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    EVAL = "EVAL"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ArtifactKind(str, Enum):
    """배포 아티팩트 종류."""

    ADAPTER = "adapter"        # LoRA adapter_model.safetensors
    MERGED = "merged"          # 머지된 full weights
    GGUF = "gguf"              # llama.cpp GGUF 포맷
    ZIP = "zip"                # 패키지 ZIP
    MODEL_CARD = "model_card"  # model_card.md


# ─────────────────────────────────────────
# HW 능력 스냅샷 (Stage 2 출력)
# ─────────────────────────────────────────


class GPUInfo(BaseModel):
    """단일 GPU 정보."""

    index: int = 0
    name: str = "unknown"
    vendor: str = "unknown"       # "nvidia" | "amd" | "apple" | "none"
    vram_gb: float = 0.0
    compute_capability: str = ""  # NVIDIA 전용 (예: "8.6")
    utilization_pct: float = 0.0
    temperature_c: float = 0.0


class CPUInfo(BaseModel):
    """CPU 정보."""

    brand: str = "unknown"
    cores_physical: int = 1
    cores_logical: int = 1
    has_avx2: bool = False
    has_avx512: bool = False


class HWCaps(BaseModel):
    """하드웨어 능력 스냅샷 — Stage 2에서 수집, 이후 모든 단계에서 참조."""

    os: str = "unknown"              # "linux" | "windows" | "darwin"
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    gpus: list[GPUInfo] = Field(default_factory=list)
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    network_mbps: float = 0.0
    # 최대 VRAM (멀티 GPU 환경에서는 단일 GPU 기준으로 계산)
    max_vram_gb: float = 0.0
    # Apple Silicon Unified Memory (darwin에서만)
    unified_memory_gb: float = 0.0
    # torch 레벨 백엔드 가용 여부
    cuda_available: bool = False
    mps_available: bool = False


# ─────────────────────────────────────────
# 파이프라인 플랜 (Stage 2 → Stage 6 입력)
# ─────────────────────────────────────────


class RecommendedPlan(BaseModel):
    """HW 프로빙 결과에서 도출된 권장 학습 플랜."""

    rule_id: str = ""                              # hw_to_plan.yaml 규칙 ID
    max_model_params_b: float = 0.0                # 최대 권장 모델 크기 (B 단위)
    strategies: list[TrainStrategy] = Field(default_factory=list)
    quantization: list[Quantization] = Field(default_factory=list)
    recommended_strategy: TrainStrategy | None = None
    recommended_quant: Quantization = Quantization.INT4
    backend: str = "cuda"                          # "cuda" | "mps" | "cpu"
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 8
    warning: str = ""                              # 경고 메시지 (CPU 전용 등)


# ─────────────────────────────────────────
# 학습 설정 (Stage 6 출력 → Stage 7 입력)
# ─────────────────────────────────────────


class LoRAConfig(BaseModel):
    """LoRA 하이퍼파라미터."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainConfig(BaseModel):
    """완전한 학습 설정 — Stage 6에서 생성."""

    strategy: TrainStrategy = TrainStrategy.QLORA
    quantization: Quantization = Quantization.INT4
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    lr_scheduler: str = "cosine"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = False
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    output_dir: str = ""


# ─────────────────────────────────────────
# 평가 결과 (Stage 8 출력)
# ─────────────────────────────────────────


class EvalResult(BaseModel):
    """평가 메트릭 집합."""

    perplexity: float | None = None
    bleu: float | None = None
    rouge1: float | None = None
    rouge2: float | None = None
    rougeL: float | None = None
    exact_match: float | None = None
    f1: float | None = None
    passed: bool = False   # 품질 기준 통과 여부
    retry_count: int = 0   # 자동 재시도 횟수


# ─────────────────────────────────────────
# 아티팩트 (Stage 9 출력)
# ─────────────────────────────────────────


class Artifact(BaseModel):
    """배포 아티팩트 메타데이터."""

    kind: ArtifactKind
    uri: str                  # 로컬 경로 또는 S3 URI
    size_bytes: int = 0
    sha256: str = ""
    created_at: datetime = Field(default_factory=_utc_now)


# ─────────────────────────────────────────
# 잡 명세 (전 스테이지의 공통 컨테이너)
# ─────────────────────────────────────────


class JobSpec(BaseModel):
    """
    파이프라인 전체 상태 컨테이너.

    Stage 1에서 생성되고 각 Stage가 자신의 출력을 필드에 채워나간다.
    서버 → ML Worker 전달 단위이자 DB jobs 테이블의 spec 컬럼이기도 하다.
    """

    # 식별자
    job_id: UUID = Field(default_factory=uuid4)
    user_id: str = ""

    # Stage 1 — 목표 정의
    ksic_section: str = ""          # KSIC 대분류 (A~U)
    ksic_division: str = ""         # KSIC 중분류 (01~99)
    task_type: TaskType = TaskType.CHAT
    description: str = ""
    language: str = "ko"

    # Stage 2 — HW 프로빙 결과
    hw_caps: HWCaps | None = None
    recommended_plan: RecommendedPlan | None = None

    # Stage 3 — 환경 설정 결과
    env_ready: bool = False
    python_version: str = ""
    cuda_version: str = ""

    # Stage 4 — 모델 선택 결과
    model_hf_repo: str = ""         # HuggingFace repo id
    model_revision: str = "main"
    model_local_path: str = ""

    # Stage 5 — 데이터 결과
    data_source_mode: DataSourceMode = DataSourceMode.UPLOAD
    dataset_path: str = ""          # 정제 완료된 데이터셋 경로
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0

    # Stage 6 — 학습 설정
    train_config: TrainConfig | None = None

    # Stage 7 — 학습 결과
    checkpoint_dir: str = ""
    train_loss: float | None = None
    eval_loss: float | None = None

    # Stage 8 — 평가 결과
    eval_result: EvalResult | None = None

    # Stage 9 — 아티팩트
    artifacts: list[Artifact] = Field(default_factory=list)

    # 상태
    state: JobState = JobState.PENDING
    current_stage: int = 0
    error_code: str = ""
    error_message: str = ""

    # 큐 우선순위 (낮을수록 먼저, PRD §10.2). 기본 5
    priority: int = 5

    # 타임스탬프
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def update_stage(self, stage: int, state: JobState = JobState.RUNNING) -> None:
        """스테이지 전이 + 타임스탬프 갱신."""
        self.current_stage = stage
        self.state = state
        self.updated_at = _utc_now()

    def to_event(self, msg: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """WebSocket 이벤트 페이로드 생성."""
        return {
            "job_id": str(self.job_id),
            "stage": self.current_stage,
            "state": self.state.value,
            "msg": msg,
            "extra": extra or {},
        }
