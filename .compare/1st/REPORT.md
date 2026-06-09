# Eden → 워크스페이스 기능 이식 리포트

## 1. 비교 개요

- **방향**: Eden(`Eden-FP-worker`, 패키지 `fp_worker`) = **기준(정답·최신)**, 워크스페이스(`Finetuning-Pipeline-worker`, 패키지 `wfp_worker`) = **수정 대상(구버전)**.
- **방법**: 각 파일을 패키지/브랜딩 토큰 정규화 후 diff. 순수 리브랜딩 hunk는 스킵, 실제 기능 추가/버그픽스만 워크스페이스 브랜딩(`WFP_`, `wfp_worker`, `Wise`, `.wfp`)으로 역매핑하여 이식.
- 워크스페이스 자체 브랜딩은 유지. Eden 브랜딩은 도입하지 않음.

## 2. 반영한 변경 목록

| 파일 | 기능 | 이식 내용 |
|---|---|---|
| `api/users.py` | admin 전용 HF 토큰 라우트 활성화 | 주석화됐던 `POST/DELETE /users/me/hf-token` 재활성화(role!=ADMIN→403, hf_/api_ 검증). `has_hf_token` 을 env/admin DB 토큰 기반 산출. |
| `db/models.py` | 카탈로그 스키마 확장 | `ModelCatalogEntry` 에 `source`/`is_public`/`local_path`/`owner_user_id` 추가. |
| `db/session.py` | 경량 DB 마이그레이션 | `_migrate_add_columns()` — init_db 에서 호출, SQLite/PG 양립 ALTER TABLE ADD COLUMN. |
| `api/admin.py` | 모델 다운로드 승인 흐름 신설 | 구 `hf/check-access`+`resume-download` 제거→ `model-download`, `env/hf-token-status`, catalog `download`/`toggle-public`/`upload` 라우트 추가. Catalog 스키마 신규 필드 + source 처리. |
| `runtime.py` | 알림/재시작/자동등록 | `hf_gated` 알림→`model_download` 알림, `revive_job_from_stage`→`retry_from_stage`, Stage9 후 `_register_finetuned_catalog`. (progress_cb 유지) |
| `api/jobs.py` | HF 토큰 자동 주입 | env HF_TOKEN > admin DB 토큰 우선순위로 spec.hf_token 주입. |
| `common/types.py` | 토큰 평문 영속화 방지(보안) | `JobSpec.hf_token` → `str \| None = Field(default=None, exclude=True)`. |
| `stage6_train_config/config_builder.py` | BF16/FP16 버그픽스 | `torch.cuda.is_bf16_supported()` 분기, bf16/fp16 동시활성 금지. |
| `stage7_train/trainer.py` | 양자화 dtype 버그픽스 | `bnb_4bit_compute_dtype` 를 config.bf16 따라 일치. |
| `server.py` | WS 견고성 | accept try/except + 연결 로깅. |
| `stage4_model/model_downloader.py` | stale 참조 정정 | `/resume-download` → `/model-download`. |
| `tests/test_e2e_api.py` | 테스트 갱신 | e12 deprecated(404)→ active(admin, 204). |

## 3. 스킵한 리브랜딩 항목
- `__init__.py`, `refresh_cli.py`(WFP_DB_URL), `api/data.py`(.wfp), `plan_recommender.py`(WFP_RULES_PATH), `collector.py`/`formatter.py`(.wfp/jobs), `packager.py`(.wfp/artifacts, WFP/Wise 문자열), `env_check.py`(WFP_TORCH_INDEX/PROJECT_DIR + 문구차) → 기능 동일.
- `test_e2e_api.py` e11, 모든 `test_stage*` → 브랜딩/노이즈 외 차이 없음.

## 4. 발견·수정한 버그
1. GradScaler/BFloat16 충돌 (config_builder+trainer): bf16/fp16 동시활성·compute_dtype 불일치 크래시 방지.
2. HF 토큰 평문 영속화: JobSpec.hf_token exclude=True 로 jobs_state.json 제외.
3. stale endpoint 참조: model_downloader 안내문 resume-download→model-download.

## 5. 보존(미반영) 결정 — 워크스페이스 고유, 버그 아님
- Stage4 progress_cb / tqdm_bridge closure-tqdm: 워크스페이스 고유 WS 진행률 기능 유지.
- `save_state(force=True)` (jobs 삭제 2곳 + admin reaper): 즉시 영속화가 더 안전 → 유지.
- runtime 상단 `import time`, save_state 주석 문구 차이: 기능 동일 → 유지.
- retry_from_stage 의 error_code/message: JobSpec str 타입에 맞춰 "" 사용(Eden None). 기능 동일.

## 6. 테스트 결과 (before → after)
- pytest (--rootdir=Finetuning-Pipeline-worker): before 114 passed/10 skipped → after 114 passed/10 skipped.
- ruff (check src tests): before 120 errors → after 129 errors (순증 +9).
  - 순증 9건은 Eden 원본 그대로 이식한 스타일 경고(E501, N814 _SL/_U/_UR/_P, 함수내부 import I001/PLC0415). 신규 기능 오류 없음. baseline 성격.
- 추가검증: 앱 import + 신규 7 라우트 등록 확인, revive_job_from_stage 제거 확인, DB 마이그레이션 스모크(model_catalog 신규 4컬럼) 통과.

## 7. 미해결 / 주의 항목
- HF 토큰 여전히 평문 DB 저장(`hf_token_encrypted`, TODO: encrypt). Eden 동일. 운영 시 Fernet/Vault 권장.
- ruff 순증 9건(스타일)은 기준 충실도 우선 미수정. 필요 시 정리 가능(기능 무영향).
- 카탈로그 다운로드/업로드 디스크 경로는 E2E 미커버(단위테스트 범위 외).
- `create_catalog` 의 `is_public=(src=='hf')` 는 Eden 원본 로직 그대로. 의도 확인 권장.
