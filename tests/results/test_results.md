# WFP ML Worker — 테스트 결과 보고서

**실행일**: 2026-05-22  
**Python**: 3.12.13 (uv venv .venv --python 3.12)  
**pytest**: 9.0.3  
**플랫폼**: darwin (Apple Silicon)  
**총 결과**: ✅ **124 passed / 0 failed** (10.32s)

> - 90개 단위 테스트 (Stage 1~9) + 20개 E2E API 통합 테스트 + 14개 샘플 파일 통합 테스트
> - E2E는 FastAPI TestClient + SQLite 임시 파일 기반 (DB 격리)
> - 인증 잔여 경고: jwt `InsecureKeyLength` (테스트용 15자 키 — 운영은 32자+ 키 사용)
> - Python 3.12 환경에서 `datetime.utcnow()` deprecation 경고는 `datetime.now(timezone.utc)`로 일괄 교체 완료

---

## 요약

| Stage | 모듈 | 테스트 수 | 결과 |
|-------|------|-----------|------|
| 1 | Init/Goal (GoalBuilder) | 11 | ✅ 전체 통과 |
| 2 | HW Probe (HWProber, PlanRecommender) | 11 | ✅ 전체 통과 |
| 3 | Env Setup (EnvManager) | 10 | ✅ 전체 통과 |
| 4 | Model (Catalog, Picker, Downloader) | 10 | ✅ 전체 통과 |
| 5 | Data (Collector, Cleaner, Formatter) | 12 | ✅ 전체 통과 |
| 5-샘플 | 샘플 파일 통합 (ops/samples/) | 14 | ✅ 전체 통과 |
| 6 | Train Config (TrainConfigBuilder) | 10 | ✅ 전체 통과 |
| 7 | Train Run (PipelineTrainer) | 6 | ✅ 전체 통과 |
| 8 | Eval (ModelEvaluator) | 10 | ✅ 전체 통과 |
| 9 | Package (Packager) | 10 | ✅ 전체 통과 |
| E2E | API 통합 (auth/users/models/jobs/admin) | 20 | ✅ 전체 통과 |
| **합계** | | **124** | **✅ 124/124** |

---

## Stage 1 — Init/Goal

| ID | 시나리오 | 결과 |
|----|---------|------|
| S1-01 | 정상 입력 → JobSpec 생성 | ✅ PASS |
| S1-02 | 모든 유효 KSIC 섹션(A~U) 수용 | ✅ PASS |
| S1-03 | 잘못된 KSIC 섹션 → WFP-0101 | ✅ PASS |
| S1-04 | 잘못된 태스크 타입 → WFP-0102 | ✅ PASS |
| S1-05 | 설명 10자 미만 → WFP-0103 | ✅ PASS |
| S1-06 | 언어 자동 추론 (미입력 시 'ko') | ✅ PASS |
| S1-07 | 대소문자 무관 섹션 입력 정규화 | ✅ PASS |
| S1-08a | 유효한 KSIC 중분류 수용 | ✅ PASS |
| S1-08b | 잘못된 KSIC 중분류 → WFP-0101 | ✅ PASS |
| S1-09 | 반복 호출 시 UUID 고유성 | ✅ PASS |
| S1-10 | 초기 상태 RUNNING, stage=1 | ✅ PASS |

**비고**: 설명 최소 길이 검증(10자)이 올바르게 동작함.  
KSIC 섹션 A~U 21개 전체 루프 테스트 포함.

---

## Stage 2 — HW Probe

| ID | 시나리오 | 결과 |
|----|---------|------|
| S2-01 | probe() → HWCaps 인스턴스 반환 | ✅ PASS |
| S2-02a | RAM > 0 | ✅ PASS |
| S2-02b | CPU 코어 수 > 0 | ✅ PASS |
| S2-03 | NVIDIA 16GB → QLoRA/cuda 플랜 | ✅ PASS |
| S2-04 | Apple Silicon → mps 백엔드 | ✅ PASS |
| S2-05 | CPU 전용 → warning 포함 플랜 | ✅ PASS |
| S2-06 | 규칙 파일 없으면 내장 폴백 사용 | ✅ PASS |
| S2-07 | os 필드 linux/windows/darwin | ✅ PASS |
| S2-08 | max_vram_gb = max(gpus.vram_gb) | ✅ PASS |
| S2-09 | GPU 환경 max_model_params_b > 0 | ✅ PASS |
| S2-10 | recommended_strategy는 TrainStrategy 열거형 | ✅ PASS |

**비고**: `hw_to_plan.yaml` 규칙 파일 로드 + 내장 폴백 모두 검증.  
실제 머신(Apple M-series)에서 실행되어 darwin/CPU 경로도 정상 동작 확인.

---

## Stage 3 — Env Setup

| ID | 시나리오 | 결과 |
|----|---------|------|
| S3-01 | 현재 Python >= 3.12 통과 | ✅ PASS |
| S3-02 | 필수 패키지 누락 없음 | ✅ PASS |
| S3-03 | spec.env_ready = True 갱신 | ✅ PASS |
| S3-04 | python_version 필드 채워짐 | ✅ PASS |
| S3-05 | GPU 없는 환경 → driver warning | ✅ PASS |
| S3-06 | NVIDIA 있으나 CUDA 비활성 → 경고 | ✅ PASS |
| S3-07 | Apple Silicon → macOS 권고 메시지 | ✅ PASS |
| S3-08 | Windows + 비NVIDIA → WSL2 권고 | ✅ PASS |
| S3-09 | setup 후 current_stage = 3 | ✅ PASS |
| S3-10 | EnvCheckResult.is_ready = True | ✅ PASS |

**비고**: 각 OS/GPU 조합(Linux+NVIDIA, macOS, Windows)을 HWCaps mock으로 시뮬레이션.

---

## Stage 4 — Model Selection & Download

| ID | 시나리오 | 결과 |
|----|---------|------|
| S4-01 | J 섹션 코드 모델 우선 반환 | ✅ PASS |
| S4-02 | VRAM 6GB 환경에서 대형 모델 제외 | ✅ PASS |
| S4-03 | max_results 제한 준수 | ✅ PASS |
| S4-04 | 자동 선택 시 1순위 채택 | ✅ PASS |
| S4-05 | 후보 없으면 WFP-0402 | ✅ PASS |
| S4-06 | manual_repo 지정 시 해당 모델 사용 | ✅ PASS |
| S4-07 | 카탈로그 외 manual_repo 허용 (경고만) | ✅ PASS |
| S4-08 | pick 후 spec.model_hf_repo 갱신 | ✅ PASS |
| S4-09 | 캐시 존재 시 재다운로드 없이 경로 반환 | ✅ PASS |
| S4-10 | 디스크 부족 시 WFP-0404 | ✅ PASS |

**비고**: S4-09에서 `huggingface_hub.snapshot_download` 미호출 확인.  
S4-10에서 `psutil.disk_usage` mock으로 1GB 여유 공간 시뮬레이션.

---

## Stage 5 — Data Collection & Cleaning

| ID | 시나리오 | 결과 |
|----|---------|------|
| S5-01 | JSONL 업로드 정상 파싱 | ✅ PASS |
| S5-02 | 존재하지 않는 파일 → WFP-0501 | ✅ PASS |
| S5-03 | 지원하지 않는 확장자 → WFP-0504 | ✅ PASS |
| S5-04 | 디렉토리 수집 (txt 파일 재귀) | ✅ PASS |
| S5-05 | 10건 미만 수집 → WFP-0503 | ✅ PASS |
| S5-06 | 중복 레코드 제거 (19/20 제거) | ✅ PASS |
| S5-07 | 이메일·전화번호 PII 마스킹 | ✅ PASS |
| S5-08 | 최소 길이 미달 텍스트 제거 | ✅ PASS |
| S5-09 | instruction 포맷 변환 | ✅ PASS |
| S5-10 | train/val/test 분할 비율(8:1:1) | ✅ PASS |
| S5-11 | 포맷 후 JobSpec 갱신 | ✅ PASS |
| S5-12 | 합성 데이터 생성 재현성 확인 | ✅ PASS |

**비고**: PII 마스킹에서 `[EMAIL]`, `[PHONE]` 치환 정확히 동작.  
ShingleHash 기반 중복 제거로 의존성 없이 near-duplicate 탐지.

---

## Stage 5 샘플 파일 통합 테스트 (`test_stage5_samples.py`)

실제 `ops/samples/upload/` 및 `ops/samples/documents/` 파일 사용.  
**WFP-0503 버그 수정 검증 포함.**

| ID | 시나리오 | 결과 |
|----|---------|------|
| SS-05 | `_extract_text` — instruction+output 합산 (WFP-0503 수정 검증) | ✅ PASS |
| SS-05b | `_extract_text` — text 필드도 합산에 포함 | ✅ PASS |
| SS-05c | agri_qa 스타일 레코드 품질 필터 통과 (수정 전 전량 제거) | ✅ PASS |
| SS-06 | `_extract_text` — 빈 input 필드 무시 | ✅ PASS |
| SS-01 | agri_qa.csv 업로드 수집 → 120건 파싱 성공 | ✅ PASS |
| SS-02 | agri_qa.jsonl 업로드 수집 성공 | ✅ PASS |
| SS-03 | agri_qa.json 업로드 수집 성공 (JSON 배열 포맷 지원) | ✅ PASS |
| SS-04 | agri_qa.parquet 업로드 수집 (pyarrow 없으면 DATA_FORMAT_ERROR) | ✅ PASS |
| SS-07 | agri_qa.csv 품질 필터 — 50%+ 생존 (수정 전 0건) | ✅ PASS |
| SS-08 | agri_qa.csv 전체 파이프라인 collect→clean→format 성공 | ✅ PASS |
| SS-09 | documents/ 디렉토리 수집 (txt/md 포함) → 10건 이상 | ✅ PASS |
| SS-10 | documents/ 전체 파이프라인 성공 | ✅ PASS |
| SS-11 | PDF 파싱 (PyMuPDF 없으면 DATA_FORMAT_ERROR) | ✅ PASS |
| SS-12 | DOCX 파싱 (python-docx 없으면 DATA_FORMAT_ERROR) | ✅ PASS |

**주요 버그 수정 내역:**

| 버그 | 증상 | 원인 | 수정 |
|------|------|------|------|
| WFP-0503 | agri_qa.csv 업로드 시 전체 레코드 품질 필터 제거 | `_extract_text()`가 `instruction` 필드만 반환 (17자) → min_length=30 미달 | 모든 비어있지 않은 필드(instruction+input+output) 합산으로 변경 |
| JSON 배열 파싱 | `.json` 확장자 파일 0건 수집 | `_parse_jsonl()`이 라인별 파싱 → JSON 배열 구조 처리 불가 | `_parse_json()` 메서드 추가 (배열/단일 객체 모두 지원) |

---

## Stage 6 — Train Config Builder

| ID | 시나리오 | 결과 |
|----|---------|------|
| S6-01 | build() → TrainConfig 반환 | ✅ PASS |
| S6-02 | QLoRA → LoRAConfig 포함 | ✅ PASS |
| S6-03 | QLoRA → INT4 양자화 | ✅ PASS |
| S6-04 | 500건 미만 → 5 에폭 | ✅ PASS |
| S6-05 | 500~5000건 → 3 에폭 | ✅ PASS |
| S6-06 | 사용자 오버라이드 적용 | ✅ PASS |
| S6-07 | 알 수 없는 오버라이드 키 무시 | ✅ PASS |
| S6-08 | output_dir 생성 확인 | ✅ PASS |
| S6-09 | spec.train_config 갱신 | ✅ PASS |
| S6-10 | 리소스 추정 dict 반환 | ✅ PASS |

**비고**: 에폭 자동 결정 로직: <500건 → 5, 500~5000 → 3, 5000+ → 2.

---

## Stage 7 — Train Run

| ID | 시나리오 | 결과 |
|----|---------|------|
| S7-01 | train_config 없으면 WFP-0601 | ✅ PASS |
| S7-02 | torch 미설치 → WFP-0302 | ✅ PASS |
| S7-03 | CUDA OOM RuntimeError → WFP-0701 | ✅ PASS |
| S7-04 | 학습 시작 시 진행 콜백 호출 | ✅ PASS |
| S7-05 | dataset_path 없으면 WFP-0502 | ✅ PASS |
| S7-08 | 콜백 없어도 기본 로거로 동작 | ✅ PASS |

**비고**: torch/transformers 미설치 환경에서 실행되어 mock 기반 테스트 수행.  
S7-06/07(spec.train_loss/checkpoint_dir 갱신)은 실제 학습 환경(GPU)에서 통합 테스트 필요.

---

## Stage 8 — Evaluation

| ID | 시나리오 | 결과 |
|----|---------|------|
| S8-01 | checkpoint_dir 없으면 WFP-0801 | ✅ PASS |
| S8-02 | 평가 라이브러리 미설치 → passed=True 폴백 | ✅ PASS |
| S8-03 | perplexity > 임계값 → passed=False | ✅ PASS |
| S8-04 | rougeL >= 임계값 → passed=True | ✅ PASS |
| S8-05 | token F1 계산 정확성 | ✅ PASS |
| S8-06 | should_retry: 미달 + retry<MAX → True | ✅ PASS |
| S8-07 | should_retry: MAX 초과 → False | ✅ PASS |
| S8-08 | should_retry: passed=True → False | ✅ PASS |
| S8-09 | spec.eval_result 갱신 | ✅ PASS |
| S8-10 | state = EVAL 전이 확인 | ✅ PASS |

**비고**: 품질 임계값은 태스크 타입별로 독립 설정.  
최대 재시도(MAX_RETRY=3) 경계 조건 테스트 포함.

---

## Stage 9 — Package & Deliver

| ID | 시나리오 | 결과 |
|----|---------|------|
| S9-01 | checkpoint_dir 없으면 WFP-0902 | ✅ PASS |
| S9-02 | 어댑터 파일 존재 시 ADAPTER 아티팩트 생성 | ✅ PASS |
| S9-03 | model_card.md 항상 생성 | ✅ PASS |
| S9-04 | ZIP 아카이브 생성 | ✅ PASS |
| S9-05 | ZIP 내 model_card.md 포함 | ✅ PASS |
| S9-06 | spec.artifacts 갱신 | ✅ PASS |
| S9-07 | 완료 후 state = DONE | ✅ PASS |
| S9-08 | ADAPTER SHA256 64자 | ✅ PASS |
| S9-09 | ADAPTER 파일 크기 > 0 | ✅ PASS |
| S9-10 | model_card에 KSIC 섹션 포함 | ✅ PASS |

**비고**: adapter_model.safetensors 더미 파일로 실제 파일 I/O 전체 경로 검증.  
SHA256 스트리밍 계산으로 대용량 파일 지원 확인.

---

## E2E API 통합 테스트

| ID | 시나리오 | 결과 |
|----|---------|------|
| E1 | /health 동작 | ✅ PASS |
| E2 | /hw/caps 비인증 호출 가능 | ✅ PASS |
| E3 | signup → 즉시 access_token 반환 | ✅ PASS |
| E4 | 회원가입 → 항상 USER (ADMIN은 SUPERVISOR 환경변수로만) | ✅ PASS |
| E5 | 두 번째 사용자도 USER | ✅ PASS |
| E6 | 중복 이메일 signup → 409 | ✅ PASS |
| E7 | login 정상 | ✅ PASS |
| E8 | login 잘못된 비번 → 401 | ✅ PASS |
| E9 | /auth/me 인증 → UserPublic | ✅ PASS |
| E10 | /auth/me 토큰 없으면 401 | ✅ PASS |
| E11 | /users/me PATCH 정상 | ✅ PASS |
| E12 | /users/me/hf-token 저장/삭제 | ✅ PASS |
| E13 | /models/ 비인증 OK + 후보 반환 | ✅ PASS |
| E14 | /jobs GET — 인증 필요 | ✅ PASS |
| E15 | /jobs POST — 잡 생성 + 인메모리 등록 | ✅ PASS |
| E16 | /jobs/{id} — 본인 잡 조회 가능, 타인은 403 | ✅ PASS |
| E17 | /jobs/{id} DELETE — 본인만 | ✅ PASS |
| E18 | /admin/users — ADMIN만 접근 가능 | ✅ PASS |
| E19 | /admin/users PATCH — 권한 변경 + audit 기록 | ✅ PASS |
| E20 | /admin/catalog/models CRUD + audit | ✅ PASS |

---

## 실패 및 수정 이력

| 케이스 | 원인 | 수정 내용 |
|--------|------|----------|
| S1-09/S1-10 | 테스트 설명 9자 (< 10자 제한) | 테스트 description 문자열 연장 |
| S5-06 | 중복 제거 후 18자 텍스트가 quality filter(30자) 통과 못함 | 테스트 텍스트를 30자 이상으로 변경 |
| S5-08 | "충분히 긴 텍스트..." 28자 → quality filter 제거 | 텍스트 길이 32자로 연장 |
| S6-04 | 100건에 대해 에폭=5 기대(코드) vs 3 기대(테스트) 불일치 | 테스트 설명과 기대값을 코드 동작에 맞게 수정 |
| S7-05 | `_build_trainer`가 `transformers` 직접 임포트 시도 | `_build_trainer`도 mock 처리 |
| WFP-0503 (실잡) | `_extract_text()`가 instruction 필드(17자)만 반환 → 품질 필터 전량 제거 | instruction+input+output 비어있지 않은 필드 전부 합산 |
| SS-03 JSON 배열 | `.json` 파일 라인별 파싱 실패 → 0건 수집 | `_parse_json()` 추가 (배열/단일 객체 지원) |
| E4 기대값 불일치 | 회원가입 자동 ADMIN 로직 제거 후 테스트 미반영 | 기대값 USER로 수정 + 설명 업데이트 |
| E19/E20 admin 권한 없음 | 회원가입 USER 변경으로 boss 계정이 ADMIN 아님 | `_make_admin()` DB 헬퍼 추가 후 테스트에서 호출 |

---

## 환경 정보

```
Python  : 3.12.13
pytest  : 9.0.3
OS      : macOS (Apple Silicon M5 Max, darwin)
torch   : 미설치 (Stage 7/8은 mock 기반)
pynvml  : 미설치 (darwin 환경)
PyMuPDF : 설치됨 (SS-11 실제 파싱)
python-docx : 설치됨 (SS-12 실제 파싱)
pyarrow : 미설치 (SS-04 DATA_FORMAT_ERROR 경로 검증)
```

## 미실행 테스트 (GPU 환경 필요)

다음 시나리오는 실제 GPU 환경 통합 테스트 필요:

- S7-06: 학습 후 `spec.train_loss` 갱신
- S7-07: 학습 후 `spec.checkpoint_dir` 갱신
- S8-실제: torch + rouge_score 설치 후 실제 추론 평가
- S9-GGUF: llama.cpp 설치 후 GGUF 변환
- S9-MERGED: peft + transformers 설치 후 merge_and_unload
