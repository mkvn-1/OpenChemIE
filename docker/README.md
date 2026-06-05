# Dockerized OpenChemIE API

This Docker setup packages the FastAPI service and both the figure and text pipelines.

The image uses `python:3.10-slim` plus CUDA-enabled PyTorch wheels. The container
does not need a full CUDA base image, but the host still needs a working NVIDIA
driver and Docker GPU runtime so `--gpus all` can expose the H100s.

The Docker build keeps `torch==2.1.2+cu121` for H100 support. Some upstream
OpenChemIE dependency metadata still declares `torch<2.0`, so the legacy Git
packages are installed with dependency resolution disabled after the core
runtime dependencies are already installed.

## Build

```bash
docker build -t openchemie-api:latest .
```

## Run On GPU

```bash
docker run --rm --gpus all -p 8000:8000 openchemie-api:latest
```

Or with Compose:

```bash
docker compose up --build
```

## Load Models

Base figure pipeline:

```bash
curl -X POST "http://localhost:8000/start?device=cuda"
```

Preload text models too:

```bash
curl -X POST "http://localhost:8000/start?device=cuda&preload_text_reactions=true&preload_text_molecules=true"
```

Select a specific GPU:

```bash
curl -X POST "http://localhost:8000/start?device=cuda:1"
```

## Health

```bash
curl "http://localhost:8000/health"
```

The response shows loaded components and CUDA memory.

## Extract PDF To ZIP

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&figure_batch_size=1&text_batch_size=8&molscribe=false&ocr=false" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o openchemie_results.zip
```

ZIP structure:

```text
metadata.json
figures/result.json
text/reactions.json
text/molecules.json
```

## Stop Models

Unload models and release PyTorch CUDA memory while keeping the API server alive:

```bash
curl -X POST "http://localhost:8000/stop"
```

## H100 Notes

The image installs CUDA 12.1 PyTorch wheels, which are appropriate for H100-class GPUs. Start with `figure_batch_size=1`, then increase after observing `/health` CUDA memory metrics.
