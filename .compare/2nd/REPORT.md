# worker 비교/QA 리포트 — 2nd pass

- 작성일: 2026-06-09
- 방향: **Eden-FP-worker = 기준(정답·최신), 워크스페이스 Finetuning-Pipeline-worker = 수정 대상**
- 기준(REF): `/Users/wise/Developer/DA(Python,R,...)/Eden-FP-worker` (패키지 `fp_worker`)
- 대상(TGT): `/Users/wise/Developer/DA(Python,R,...)/Finetuning-Pipeline-worker` (패키지 `wfp_worker`)
- 1st 리포트: `.compare/1st/REPORT.md`
- 네이밍 규칙: fp_worker→wfp_worker, Worktro→Wise, Eden-FP-*→Finetuning-Pipeline-*, FP_→WFP_ (리브랜딩 차이는 버그 아님, 미이식)

## 1. 비교 개요 (2nd)

1st에서 admin HF토큰/카탈로그/알림 기능을 이식·커밋함. 2nd는 현재 디스크 상태를 Eden과 **재diff**하여 잔여 기능 격차를 정밀 점검. **두 부류의 잔여 차이 발견 → 모두 이식.**

## 2. 1st 대비 잔여 차이 (2nd에서 발견)

### A. 학습 안정화 버그픽스 (소규모 데이터셋 크래시)
1st에 누락된 Eden의 trainer/config 안정화가 워크스페이스에 미반영 상태였음.

### B. **모델 포맷 변환 백엔드 전체 누락 (Critical)**
프론트엔드(2nd)는 변환 UI(`/jobs/{id}/convert`, `/convert/status`, 카탈로그 파일 다운로드)를 이미 호출하는데 **워크스페이스 worker에는 변환 백엔드가 통째로 없었음** → 런타임 404 불일치. Eden에는 `stage9_package/converters.py`, jobs.py 변환 엔드포인트, `model_format` 카탈로그 필드가 존재. 불균등 이식으로 생긴 교차 레포 회귀.

## 3. 추가 이식/수정 내역 (이번 pass)

| 파일 | 변경 | 내용 |
|------|------|------|
| `stage6_train_config/config_builder.py` | 수정 | `logging_steps`를 전체 스텝의 1/5(최대 10) 이하로 캡 — 소규모 런에서 손실 그래프가 비는 문제 해결 |
| `stage7_train/trainer.py` | 수정 | val.jsonl 비었을 때 validation split 생략(`load_dataset` 크래시 방지), `eval_strategy`/`load_best_model_at_end`를 val 유무로 조건화, `loss`/`train_loss`·`eval_loss` 키 모두 포착 + `spec.train_loss/eval_loss` 실시간 반영 |
| `stage9_package/converters.py` | **신규(339L)** | `convert()`, `tooling_status()`, `ToolingMissingError`, `ConvertFailedError` — merged/gguf/mlx/pytorch/onnx 변환 |
| `stage9_package/__init__.py` | 수정 | 변환 심볼 export 추가 |
| `db/models.py` | 수정 | `model_catalog.model_format` 컬럼 추가(String(16), default `safetensors`) |
| `db/session.py` | 수정 | 경량 마이그레이션에 `model_format` 추가 — 기존 SQLite DB 자동 보정 |
| `api/jobs.py` | 수정 | `_converting` 상태, `_do_convert`, `POST /{job_id}/convert`(ConvertResult), `GET /{job_id}/convert/status`(ConvertStatusResult), 다운로드 model_format 처리. import는 `wfp_worker`로 역매핑 |
| `api/models_api.py` | 수정 | 파인튜닝 가능 모델만 베이스 노출하는 model_format 게이팅 |
| `common/types.py` | 수정 | ConvertFormat/ArtifactKind/Convert* 모델 |
| `pyproject.toml` | 수정 | `[project.optional-dependencies].convert` extra 추가 (ops `make install-convert-tools` 가 설치) |
| `tests/test_stage9_converters.py` | **신규** | 변환기 단위 테스트(+5) |

모든 import/branding은 `wfp_worker`/`WFP`/`Wise`로 역매핑. Eden 브랜딩 유출 없음.

## 4. 스킵한 리브랜딩 항목

- 패키지/모듈 `fp_worker`(대상은 `wfp_worker`), 에러코드 `FP_*`(대상 `WFP_*`), 제품명 Worktro(대상 Wise) 차이 — 의도된 정체성, 미이식.

## 5. 테스트 결과 (before → after)

| 검증 | before(2nd 시작) | after |
|------|------|-------|
| `python -c "import wfp_worker.server"` | OK | OK |
| pytest | 114 passed / 10 skipped | **119 passed / 10 skipped** (변환 테스트 +5) |
| ruff check src | ~120 errors(style noise) | 98 (style noise, 기능 오류 아님) |

## 6. 미해결 / 주의 항목

- **변환 도구 런타임 의존**: 실제 gguf/mlx/onnx 변환은 외부 도구 필요(llama.cpp, mlx-lm 등). `pyproject [convert]` extra + ops `make install-convert-tools`로 설치, `docker/.env`의 `WFP_LLAMA_CPP_DIR`/`WFP_CONVERT_TIMEOUT` 설정 필요. 미설치 시 `ToolingMissingError`로 graceful 처리됨(엔드포인트 자체는 정상).
- **HF 토큰 평문 저장**(`# TODO: encrypt`) — Eden과 동일, 프로덕션 암호화(Fernet/Vault) 권장. (1st부터 이월)
- ruff style noise(E501 등)는 Eden 기준 코드에도 동일 존재 — 회귀 아니므로 미수정.
