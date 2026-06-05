FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements-docker.txt requirements-docker.txt
RUN python -m pip install -r requirements-docker.txt

COPY requirements-openchemie-git.txt requirements-openchemie-git.txt
RUN python -m pip install --no-deps -r requirements-openchemie-git.txt

COPY . .
RUN python -m pip install --no-deps -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.figure_pipeline_api:app", "--host", "0.0.0.0", "--port", "8000"]
