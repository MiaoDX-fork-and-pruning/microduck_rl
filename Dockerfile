FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG UV_VERSION=0.8.19

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/microduck_rl/.venv \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install 3.12

WORKDIR /opt/microduck_rl
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN uv sync --frozen --no-dev --offline --python 3.12

# Training writes logs relative to cwd. Keep image contents immutable and use
# /outputs as the local bind mount or CloudML JuiceFS output mount.
WORKDIR /outputs
ENTRYPOINT ["/opt/microduck_rl/.venv/bin/train"]
