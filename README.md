# WFP ML Worker

**Wise Fine-tuning Pipeline** — ML 작업 실행 엔진.

사용자가 분야(KSIC 카테고리)를 선택하면 데이터 수집 → 정제 → 학습 → 평가 → 패키징까지
9단계 파이프라인을 자동으로 수행한다.

---

## 목차

1. [요구 사항](#1-요구-사항)
2. [환경 설정](#2-환경-설정)
3. [서버 실행](#3-서버-실행)
4. [파이프라인 단계별 사용법](#4-파이프라인-단계별-사용법)
5. [API 엔드포인트](#5-api-엔드포인트)
6. [HW 규칙 파일 수정](#6-hw-규칙-파일-수정)
7. [테스트 실행](#7-테스트-실행)
8. [디렉토리 구조](#8-디렉토리-구조)
9. [에러 코드 참조](#9-에러-코드-참조)

---

## 1. 요구 사항

| 항목        | 최소 버전 | 비고               |
| ----------- | --------- | ------------------ |
| Python      | 3.12      | 3.12도 지원        |
| uv          | 최신      | 패키지 관리자      |
| CUDA (선택) | 12.1+     | NVIDIA GPU 학습 시 |
| ROCm (선택) | 5.6+      | AMD GPU 학습 시    |

> **CPU 전용 환경**에서도 서버 실행 및 데이터 처리까지는 가능.
> 실제 모델 학습에는 GPU 권장.

---

## 2. 환경 설정

### 2-1. uv 설치 (미설치 시)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew (macOS)
brew install uv
```

### 2-2. 가상 환경 생성 및 기본 의존성 설치

```bash
cd Finetuning-Pipeline-worker

# Python 3.12 기반 가상환경 생성
uv venv .venv --python 3.12

# 기본 패키지 설치 (서버 실행, 데이터 처리)
uv pip install -e "." --python .venv/bin/python
```

### 2-3. 학습 패키지 설치 (GPU 환경 — 선택)

```bash
# NVIDIA CUDA 환경
uv pip install -e ".[train]" --python .venv/bin/python

# Apple Silicon (MPS)
uv pip install -e ".[train]" --python .venv/bin/python
# bitsandbytes 미지원 → unsloth 또는 mlx-lm 별도 설치 필요
# pip install mlx-lm  (Apple Silicon 전용)
```

### 2-4. 개발/테스트 의존성 설치

```bash
uv pip install -e ".[dev]" --python .venv/bin/python
```

### 2-5. 환경 변수 설정

`.env.example`을 복사하여 `.env`를 생성하고 값을 채운다.

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SUPERVISOR_ID` | (필수 권장) | 관리자 계정 이메일 — 서버 시작 시 자동 생성/보장 |
| `SUPERVISOR_PASSWORD` | (필수 권장) | 관리자 계정 비밀번호 |
| `JWT_SECRET` | `dev-jwt-secret-CHANGE-ME` | JWT 서명 키 — 운영에서 반드시 교체 |
| `JWT_ACCESS_EXPIRE_MIN` | `10080` (7일) | 토큰 만료 시간(분) |
| `HF_TOKEN` | (선택) | 비공개 HF 모델 다운로드 시 필요 |
| `DATABASE_URL` | `sqlite://~/.wfp/wfp.db` | SQLite(dev) 또는 `postgresql://...` |

> 회원가입으로 가입한 사용자는 항상 USER. ADMIN 계정은 `SUPERVISOR_ID`/`SUPERVISOR_PASSWORD`로만 생성.

### 2-6. 가상환경 활성화

```bash
source .venv/bin/activate
```

---

## 3. 서버 실행

### 개발 모드

```bash
# 권장 — Finetuning-Pipeline-ops Makefile (.env 자동 로드)
make dev-worker

# 또는 직접 실행 (사전에 export 또는 .env 수동 소스 필요)
cd Finetuning-Pipeline-worker
source .venv/bin/activate
uvicorn wfp_worker.server:app --reload --host 0.0.0.0 --port 8000
```

### 프로덕션 모드

```bash
uvicorn wfp_worker.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --log-level info
```

### 헬스체크 확인

```bash
curl http://localhost:8000/health
# {"status": "ok", "jobs": 0}
```

### 현재 HW 스냅샷 확인

```bash
curl http://localhost:8000/hw/caps | python3 -m json.tool
```

### Docker 기반 실행 (권장)

`Finetuning-Pipeline-ops/docker/`에 통합 compose 파일이 있다. Makefile이 호스트의 GPU 유무를 자동 감지해 CPU/GPU 모드를 선택한다.

```bash
# Finetuning-Pipeline-ops 저장소에서
make docker-dev          # GPU 자동 감지
make docker-dev-cpu      # CPU 강제
make docker-dev-gpu      # GPU 강제 (nvidia 런타임 필요)
```

#### 워커 단독 빌드/실행

```bash
# CPU 이미지 (multi-stage python:3.12-slim, ~1.5GB)
docker build -t wfp-worker:latest .
docker run -p 8000:8000 -v ~/.wfp:/home/wfp/.wfp wfp-worker:latest

# GPU 이미지 (python:3.12-slim + PyTorch +cu121 휠로 CUDA 런타임 번들, ~5GB)
docker build -f Dockerfile.gpu -t wfp-worker:gpu .
docker run --gpus all -p 8000:8000 -v ~/.wfp:/home/wfp/.wfp wfp-worker:gpu
```

#### GPU 이미지 설계 노트

`Dockerfile.gpu`는 `nvidia/cuda` 베이스 이미지를 **사용하지 않는다**. 대신:

- 베이스: `python:3.12-slim-bookworm` (~50MB)
- CUDA 런타임: PyTorch `+cu121` 휠에 번들된 `libcudart`/`libcublas`/`libcudnn` 등
- 호스트 요구사항: NVIDIA 드라이버 + NVIDIA Container Toolkit (`--gpus all`)
- 장점: apt-get 단계 최소화 (tzdata 대화형 프롬프트 회피), 멀티 스테이지 빌드로 최종 이미지 슬림

자세한 Docker 가이드: [`../Finetuning-Pipeline-ops/docker/README.md`](../Finetuning-Pipeline-ops/docker/README.md)

---

## 4. 파이프라인 단계별 사용법

각 스테이지를 Python 코드에서 직접 호출하거나 REST API로 실행 가능.

### 전체 파이프라인 — REST API

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-001",
    "ksic_section": "J",
    "ksic_division": "62",
    "task_type": "code",
    "description": "소프트웨어 개발 지원 AI 어시스턴트 모델",
    "language": "ko",
    "data_source_mode": "upload",
    "data_source_path": "/path/to/data.jsonl"
  }'
# {"job_id": "...", "state": "RUNNING"}
```

### 잡 상태 조회

```bash
curl http://localhost:8000/jobs/{job_id}
```

### 실시간 진행 스트림 (WebSocket)

```python
import asyncio
import websockets
import json

async def watch():
    uri = "ws://localhost:8000/ws/{job_id}"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            print(json.loads(msg))

asyncio.run(watch())
```

---

### 스테이지별 단독 실행 (Python)

#### Stage 1 — 목표 정의

```python
from wfp_worker.stage1_init import GoalBuilder

spec = GoalBuilder().build(
    ksic_section="J",       # KSIC 대분류 (A~U)
    task_type="code",       # chat | qa | summary | classification | code
    description="소프트웨어 개발 지원 모델입니다.",
    ksic_division="62",     # 선택 — KSIC 중분류
    language="ko",
    user_id="user-001",
)
print(spec.job_id)
```

#### Stage 2 — HW 프로빙

```python
from wfp_worker.stage2_hw_probe import HWProber, PlanRecommender

caps = HWProber().probe()
print(f"GPU: {caps.gpus}, VRAM: {caps.max_vram_gb}GB, RAM: {caps.ram_gb}GB")

plan = PlanRecommender().recommend(caps)
print(f"권장 전략: {plan.recommended_strategy}, 최대 모델: {plan.max_model_params_b}B")

spec.hw_caps = caps
spec.recommended_plan = plan
```

#### Stage 3 — 환경 검사

```python
from wfp_worker.stage3_env_setup import EnvManager

result = EnvManager().setup(spec)
if result.driver_warnings:
    for w in result.driver_warnings:
        print(f"경고: {w}")
print(f"환경 준비: {result.is_ready}")
```

#### Stage 4 — 모델 선택 및 다운로드

```python
from wfp_worker.stage4_model import ModelPicker, ModelDownloader, ModelCatalog

# 후보 목록 확인
catalog = ModelCatalog()
candidates = catalog.get_candidates(
    ksic_section="J",
    max_vram_gb=caps.max_vram_gb,
    language="ko",
)
for m in candidates:
    print(f"{m.hf_repo} ({m.params_b}B, VRAM>={m.hw_min_vram_gb}GB)")

# 자동 선택
repo = ModelPicker().pick(spec)

# 수동 선택
# repo = ModelPicker().pick(spec, manual_repo="Qwen/Qwen2.5-Coder-7B-Instruct")

# 다운로드 (캐시: ~/.wfp/models/)
path = ModelDownloader().download(spec)
print(f"모델 경로: {path}")
```

#### Stage 5 — 데이터 수집·정제

```python
from wfp_worker.stage5_data import DataCollector, DataCleaner, DataFormatter
from wfp_worker.common.types import DataSourceMode
import json
from pathlib import Path

# 수집 (모드 A: 파일 업로드)
raw_path = DataCollector().collect(
    spec=spec,
    source_mode=DataSourceMode.UPLOAD,
    source_path="/path/to/data.jsonl",
)

# 수집 (모드 B: 디렉토리)
# raw_path = DataCollector().collect(
#     spec=spec,
#     source_mode=DataSourceMode.DIRECTORY,
#     source_path="/path/to/docs/",
# )

# 원시 레코드 로드
raw_records = [json.loads(l) for l in Path(raw_path).read_text().splitlines() if l]

# 정제
from wfp_worker.stage5_data.cleaner import CleanConfig
cfg = CleanConfig(target_language="ko", enable_pii=True)
cleaned, stats = DataCleaner(cfg).clean(raw_records)
print(f"정제 결과: {stats.input_count} → {stats.output_count}건")

# 포맷 변환 + 분할
dataset_dir = DataFormatter().format_and_split(
    spec=spec,
    records=cleaned,
    train_ratio=0.8,
    val_ratio=0.1,
)
print(f"데이터셋: {spec.num_train}/{spec.num_val}/{spec.num_test} (train/val/test)")
```

#### Stage 6 — 학습 설정

```python
from wfp_worker.stage6_train_config import TrainConfigBuilder

builder = TrainConfigBuilder()
config = builder.build(spec)

# 사용자 오버라이드 (고급)
# config = builder.build(spec, override={"num_epochs": 5, "learning_rate": 1e-4})

# 리소스 예상량 확인
est = builder.estimate_resources(spec)
print(f"예상 {est['estimated_hours']}시간, VRAM {est['estimated_vram_gb']}GB")
```

#### Stage 7 — 학습 실행 (GPU 필요)

```python
from wfp_worker.stage7_train import PipelineTrainer

def on_progress(job_id, stage, msg, extra):
    print(f"[Step {extra.get('step','')}] loss={extra.get('loss','')}")

checkpoint = PipelineTrainer(progress_cb=on_progress).train(spec)
print(f"체크포인트: {checkpoint}")
```

#### Stage 8 — 평가 (GPU 필요)

```python
from wfp_worker.stage8_eval import ModelEvaluator

evaluator = ModelEvaluator()
result = evaluator.evaluate(spec)
print(f"Perplexity: {result.perplexity}, ROUGE-L: {result.rougeL}")
print(f"품질 기준 통과: {result.passed}")

if evaluator.should_retry(result):
    print("자동 재시도 권고 — epoch 수를 늘려 재학습")
```

#### Stage 9 — 패키징

```python
from wfp_worker.stage9_package import Packager

artifacts = Packager().package(
    spec,
    create_merged=False,  # 베이스+어댑터 병합 (선택)
    create_gguf=False,    # GGUF 변환 (llama.cpp 필요)
    create_zip=True,
)

for a in artifacts:
    print(f"{a.kind.value}: {a.uri} ({a.size_bytes // 1024}KB)")
```

---

## 5. API 엔드포인트

### 공통 / 사용자

| 메서드   | 경로         | 설명                       |
| -------- | ------------ | -------------------------- |
| `GET`    | `/health`    | 서버 헬스체크              |
| `GET`    | `/hw/caps`   | 현재 HW 스냅샷             |
| `POST`   | `/auth/signup` | 회원가입 + 토큰 발급 (USER) |
| `POST`   | `/auth/login` | 로그인                     |
| `GET`    | `/auth/me`   | 현재 사용자                |
| `POST`   | `/jobs`      | 잡 생성 및 파이프라인 실행 |
| `GET`    | `/jobs/{id}` | 잡 상태 조회               |
| `DELETE` | `/jobs/{id}` | 잡 취소                    |
| `WS`     | `/ws/{id}`   | 실시간 진행 스트림         |

### 관리자 (PRD §10 — ADMIN 권한)

| 메서드   | 경로         | 설명                       | PRD |
| -------- | ------------ | -------------------------- | --- |
| `GET`    | `/admin/stats` | 대시보드 카드 (사용자/잡/타이밍/GPU breakdown) | §10.1 |
| `GET`    | `/admin/resources` | GPU/큐/디스크 현황 | §10.1 |
| `GET`    | `/admin/jobs` | 전체 잡 (필터 지원) | §10.2 |
| `GET`    | `/admin/jobs/{id}` | 잡 상세 + 이벤트 로그 | §10.2 |
| `POST`   | `/admin/jobs/{id}/cancel` | 강제 취소 | §10.2 |
| `PATCH`  | `/admin/jobs/{id}/priority` | 우선순위 변경 (1~9) | §10.2 |
| `POST`   | `/admin/reaper/run` | Reaper 수동 트리거 (heartbeat 끊김/디스크 회수) | §10.3 |
| CRUD     | `/admin/catalog/models[/{id}]` | 모델 카탈로그 | §10.4 |
| CRUD     | `/admin/data-sources[/{id}]` | 데이터 소스 화이트리스트 | §10.4 |
| CRUD     | `/admin/presets[/{id}]` | KSIC/태스크 프리셋 | §10.4 |
| `GET`    | `/admin/users` | 사용자 목록 | |
| `PATCH`  | `/admin/users/{id}` | 권한/활성 토글 | |
| `GET`    | `/admin/audit` | 감사 로그 (필터: actor_email/action/date) | §10.5 |
| `GET`    | `/admin/env/status` | 호스트 환경 진단 | |
| `POST`   | `/admin/env/install` | 학습 패키지 자동 설치 | |

### POST /jobs 요청 파라미터

```json
{
  "user_id": "string",
  "ksic_section": "J", // 필수: A~U
  "ksic_division": "62", // 선택: 01~99
  "task_type": "code", // chat | qa | summary | classification | code
  "description": "모델 용도 설명 (10자 이상)",
  "language": "ko", // ko | en | ja | zh
  "model_hf_repo": "", // 빈 문자열 = 자동 선택
  "data_source_mode": "upload", // upload | dir | web | synth
  "data_source_path": "/path/...",
  "keywords": [], // web 모드 시 검색 키워드
  "train_override": {
    // 고급 사용자 하이퍼파라미터 오버라이드
    "num_epochs": 3,
    "learning_rate": 2e-4
  }
}
```

### Swagger UI

서버 실행 후 브라우저에서 확인:

```
http://localhost:8000/docs
```

---

## 6. HW 규칙 파일 수정

`../Finetuning-Pipeline-ops/rules/hw_to_plan.yaml`에서 GPU → 플랜 매핑 규칙을 수정한다.
코드 재시작 없이 파일만 수정해도 다음 잡 실행 시 반영된다.

```yaml
rules:
  - id: my_custom_rule
    condition:
      gpu_vendor: nvidia
      vram_gb_min: 20
      vram_gb_max: 24
    plan:
      max_model_params_b: 13
      strategies: [qlora]
      quantization: [int4]
      recommended_strategy: qlora
      recommended_quant: int4
      max_seq_length: 2048
      batch_size: 2
      gradient_accumulation: 8
```

**조건 키**:

| 키                      | 설명                                   |
| ----------------------- | -------------------------------------- |
| `gpu_vendor`            | `nvidia` \| `amd` \| `apple` \| `none` |
| `vram_gb_min`           | NVIDIA VRAM 최솟값 (GB)                |
| `vram_gb_max`           | NVIDIA VRAM 최댓값 (GB)                |
| `unified_memory_gb_min` | Apple Unified Memory 최솟값            |
| `ram_gb_min`            | CPU-only 환경 RAM 최솟값               |

---

## 7. 테스트 실행

```bash
# 전체 테스트
.venv/bin/python -m pytest tests/ -v

# 특정 스테이지만
.venv/bin/python -m pytest tests/test_stage2_hw_probe.py -v

# 커버리지 포함
.venv/bin/python -m pytest tests/ --cov=wfp_worker --cov-report=term-missing

# 빠른 실패 (첫 번째 실패 시 중단)
.venv/bin/python -m pytest tests/ -x
```

테스트 결과 보고서: [tests/results/test_results.md](tests/results/test_results.md)

### 현재 테스트 현황

```
90 passed, 0 failed  (Python 3.12, macOS Apple Silicon)
```

> Stage 7/8 실제 학습·평가 테스트는 GPU + torch 설치 환경에서 별도 실행 필요.

---

## 8. 디렉토리 구조

```
Finetuning-Pipeline-worker/
├── pyproject.toml                  # 의존성, ruff, pytest 설정
├── src/
│   └── wfp_worker/
│       ├── common/
│       │   ├── types.py            # JobSpec, HWCaps, TrainConfig 등 공유 타입
│       │   ├── errors.py           # WFP-XXYY 에러 코드
│       │   └── logger.py           # 구조화 JSON 로거
│       ├── stage1_init/
│       │   └── goal_builder.py     # KSIC 검증 → JobSpec 생성
│       ├── stage2_hw_probe/
│       │   ├── prober.py           # psutil/pynvml/system_profiler 수집
│       │   └── plan_recommender.py # hw_to_plan.yaml 규칙 평가
│       ├── stage3_env_setup/
│       │   └── env_manager.py      # Python/패키지/드라이버 검증
│       ├── stage4_model/
│       │   ├── model_catalog.py    # KSIC별 모델 카탈로그
│       │   ├── model_picker.py     # HW 필터 + 자동/수동 선택
│       │   └── model_downloader.py # HuggingFace Hub 다운로드
│       ├── stage5_data/
│       │   ├── collector.py        # JSONL/CSV/Dir/Web 수집
│       │   ├── cleaner.py          # dedup/langfilter/PII 정제
│       │   └── formatter.py        # instruction 변환 + 분할
│       ├── stage6_train_config/
│       │   └── config_builder.py   # HW 플랜 → TrainConfig 자동 생성
│       ├── stage7_train/
│       │   └── trainer.py          # SFTTrainer + LoRA/QLoRA 실행
│       ├── stage8_eval/
│       │   └── evaluator.py        # perplexity/ROUGE/F1 + 재시도 판정
│       ├── stage9_package/
│       │   └── packager.py         # adapter/model_card/ZIP 패키징
│       └── server.py               # FastAPI + WebSocket 서버
├── tests/
│   ├── conftest.py                 # 공통 픽스처
│   ├── test_stage1_init.py
│   ├── ...
│   ├── test_stage9_package.py
│   └── results/
│       └── test_results.md
# HW 규칙: ../Finetuning-Pipeline-ops/rules/hw_to_plan.yaml (별도 저장소)
```

### 주요 캐시 경로

| 경로                            | 내용                  |
| ------------------------------- | --------------------- |
| `~/.wfp/models/`                | HuggingFace 모델 캐시 |
| `~/.wfp/jobs/{job_id}/raw/`     | 원시 수집 데이터      |
| `~/.wfp/jobs/{job_id}/dataset/` | 정제·분할된 데이터셋  |
| `~/.wfp/checkpoints/{job_id}/`  | 학습 체크포인트       |
| `~/.wfp/artifacts/{job_id}/`    | 최종 패키지 아티팩트  |

---

## 9. 에러 코드 참조

| 코드       | 스테이지 | 원인                            |
| ---------- | -------- | ------------------------------- |
| `WFP-0101` | 1        | 유효하지 않은 KSIC 섹션/중분류  |
| `WFP-0102` | 1        | 유효하지 않은 태스크 타입       |
| `WFP-0103` | 1        | 설명 10자 미만                  |
| `WFP-0201` | 2        | HW 프로빙 기본 정보 수집 실패   |
| `WFP-0203` | 2        | 매칭되는 HW 규칙 없음           |
| `WFP-0301` | 3        | Python 버전 미달 (< 3.12)       |
| `WFP-0302` | 3        | 필수 패키지 미설치              |
| `WFP-0401` | 4        | HuggingFace 다운로드 실패       |
| `WFP-0402` | 4        | HW 조건에 맞는 모델 없음        |
| `WFP-0404` | 4        | 디스크 공간 부족                |
| `WFP-0405` | 4        | HF 토큰 없음 (비공개 모델)      |
| `WFP-0501` | 5        | 업로드 파일 없음                |
| `WFP-0503` | 5        | 수집 데이터 10건 미만           |
| `WFP-0504` | 5        | 지원하지 않는 파일 형식         |
| `WFP-0601` | 6        | 학습 설정 생성 실패 (플랜 없음) |
| `WFP-0701` | 7        | GPU OOM                         |
| `WFP-0702` | 7        | 학습 중 크래시                  |
| `WFP-0801` | 8        | 평가 실행 실패                  |
| `WFP-0803` | 8        | 품질 기준 미달                  |
| `WFP-0901` | 9        | GGUF 변환 실패                  |
| `WFP-0902` | 9        | 아티팩트 저장 실패              |

---

## 라이선스

내부 사용 전용. 베이스 모델 각각의 라이선스를 확인하고 준수할 것.
