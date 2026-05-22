# =============================================================================
# WFP ML Worker — Python 3.12 + FastAPI + uv
# Multi-stage 빌드로 최종 이미지 크기 최소화.
# GPU 학습은 docker-compose.gpu.yaml에서 nvidia 런타임 + CUDA 베이스 이미지 사용.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Builder: uv로 의존성 휠 빌드
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# 빌드 도구 (psutil/numpy 컴파일에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (공식 단일 바이너리)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /build

# 의존성 메타데이터 먼저 복사 (캐시 효율)
COPY pyproject.toml ./

# 가상환경 + 의존성 설치 (기본 + train CPU/MPS)
# torch는 CPU index 사용 (큰 CUDA wheel 회피 — Dockerfile.gpu가 CUDA 담당)
RUN uv venv /opt/venv --python 3.12 && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "."

# 소스 복사 후 정식 설치 + train extras (CPU 학습 가능)
COPY src/ ./src/
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[train]"

# -----------------------------------------------------------------------------
# Stage 2 — Runtime: 슬림 이미지 + 가상환경만 복사
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# 런타임 의존성 (libgomp1=OpenMP, curl=헬스체크용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r wfp && useradd -r -g wfp -d /home/wfp -m wfp

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    WFP_WORKER_HOST=0.0.0.0 \
    WFP_WORKER_PORT=8000

# 빌더에서 가상환경만 가져옴
COPY --from=builder /opt/venv /opt/venv

# 워크 디렉토리 + 캐시 경로 사전 생성 (~/.wfp는 wfp 유저 홈에 마운트)
# /data는 SQLite DB 영속화 볼륨 (compose volume `wfp_data`)
WORKDIR /app
RUN mkdir -p /home/wfp/.wfp/{models,jobs,checkpoints,artifacts} /data && \
    chown -R wfp:wfp /home/wfp/.wfp /app /data

# ops/rules/hw_to_plan.yaml은 호스트에서 마운트 (또는 이미지에 포함)
COPY --chown=wfp:wfp src/ /app/src/

USER wfp

EXPOSE 8000

# 헬스체크 — /health 엔드포인트
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# tini로 PID 1 시그널 처리 (Ctrl+C 정상 종료)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "wfp_worker.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
