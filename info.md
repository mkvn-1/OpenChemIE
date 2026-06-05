# OpenChemIE Docker + GPU API Guide

This repo now has a Dockerized FastAPI service for OpenChemIE PDF extraction. It supports figure extraction and text-based extraction, and can return JSON or a ZIP with separate output folders.

## What This API Does

The API accepts a PDF and extracts chemistry information using two paths:

- Figure pipeline: finds reaction figures in the PDF and runs OpenChemIE/RxnScribe on them.
- Text pipeline: uses Poppler `pdftotext` through the local compatibility wrapper and runs text molecule/reaction extraction when requested.

For ZIP output, the response is organized like this:

```text
metadata.json
figures/result.json
text/reactions.json
text/molecules.json
```

Only the folders for enabled extraction parts are included.

## Docker Requirements

On a GPU server, install:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit

Check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If this command fails, fix NVIDIA Container Toolkit before running this repo.

## Build Docker Image

From the repo root:

```bash
docker build -t openchemie-api:test .
```

The image uses:

```text
python:3.10-slim-bookworm
```

It is still large because CUDA PyTorch, torchvision, RDKit, OpenCV, EasyOCR, and the OpenChemIE model dependencies are large.

## Run Docker Container

Run with GPU access:

```bash
docker run -d --gpus all --name openchemie-api-test -p 8000:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

If port `8000` is already busy, use another host port:

```bash
docker run -d --gpus all --name openchemie-api-test -p 8001:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

Then use:

```text
http://localhost:8001
```

Stop and remove the container:

```bash
docker rm -f openchemie-api-test
```

View logs:

```bash
docker logs openchemie-api-test
```

## Docker Compose

Start with Compose:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

## GPU Selection

Docker GPU access is controlled by:

```bash
--gpus all
```

or in Compose:

```yaml
NVIDIA_VISIBLE_DEVICES: all
```

Inside the API, `/start?device=cuda` asks PyTorch to use CUDA. PyTorch will use the first visible CUDA GPU as `cuda:0`.

Important behavior:

- If Docker exposes one GPU, the API uses that GPU.
- If Docker exposes multiple GPUs, PyTorch defaults to `cuda:0`.
- `cuda:0` means the first GPU visible inside the container, not always physical GPU 0 on the host.
- To choose a specific GPU manually, expose only that GPU to the container.

Example: expose only GPU 1:

```bash
docker run -d --gpus '"device=1"' --name openchemie-api-test -p 8000:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

For your server with two H100 GPUs, the safest production pattern is to choose the GPU at container launch based on free memory from `nvidia-smi`, then expose only that GPU. The current API does not rebalance models across GPUs after startup. It loads onto the requested PyTorch device.

Check host GPU memory:

```bash
nvidia-smi
```

Run on the selected GPU:

```bash
docker run -d --gpus '"device=0"' --name openchemie-api-test -p 8000:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

or:

```bash
docker run -d --gpus '"device=1"' --name openchemie-api-test -p 8000:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

## Endpoint Summary

Base URL:

```text
http://localhost:8000
```

If you mapped host port `8001`, use:

```text
http://localhost:8001
```

### Health

Shows server state, loaded model components, CUDA availability, CUDA device count, and memory stats.

```bash
curl http://localhost:8000/health
```

Example fields:

```json
{
  "loaded": true,
  "status": "ready",
  "loaded_components": ["pdfparser", "rxnscribe"],
  "cuda_available": true,
  "cuda_device_count": 1
}
```

### Start

Downloads and loads the requested models. Use this once before extraction.

Basic GPU load:

```bash
curl -X POST "http://localhost:8000/start?device=cuda"
```

Load only the base figure pipeline:

```bash
curl -X POST "http://localhost:8000/start?device=cuda&preload_molscribe=false&preload_moldet=false&preload_coref=false&preload_text_reactions=false&preload_text_molecules=false"
```

Optional preload flags:

```text
preload_molscribe=true
preload_moldet=true
preload_coref=true
preload_text_reactions=true
preload_text_molecules=true
```

On H100, these are more practical than on a small laptop GPU, but load only what you need.

### Stop

Unloads models from the API state and clears PyTorch CUDA cache as much as possible.

```bash
curl -X POST "http://localhost:8000/stop"
```

After this, `/health` should show:

```json
{
  "loaded": false,
  "status": "stopped"
}
```

Some small CUDA context memory can remain while the Python process is alive. That is normal.

### Extract Figures

Runs figure extraction only.

JSON output:

```bash
curl -X POST "http://localhost:8000/extract-figures?batch_size=1&num_pages=1&molscribe=false&ocr=false" -F "pdf=@example/acs.joc.2c00749.pdf" -o result.json
```

ZIP output:

```bash
curl -X POST "http://localhost:8000/extract-figures?response_format=zip&batch_size=1&num_pages=1&molscribe=false&ocr=false" -F "pdf=@example/acs.joc.2c00749.pdf" -o figures.zip
```

Useful parameters:

```text
response_format=json|zip
batch_size=1
num_pages=1
molscribe=false
ocr=false
split_large_figures=true
deduplicate_figures=true
include_overlays=true
```

Use `molscribe=true` and `ocr=true` only when you need richer output.

### Extract PDF

Runs combined figure + text extraction.

First-page smoke test with separate `figures/` and `text/` folders:

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&num_pages=1&figure_batch_size=1&text_batch_size=8&include_figures=true&include_text_reactions=false&include_text_molecules=true&molscribe=false&ocr=false" -F "pdf=@example/acs.joc.2c00749.pdf" -o openchemie_first_page.zip
```

Full PDF extraction:

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&figure_batch_size=1&text_batch_size=8&include_figures=true&include_text_reactions=true&include_text_molecules=true&molscribe=false&ocr=false" -F "pdf=@example/acs.joc.2c00749.pdf" -o openchemie_results.zip
```

Useful parameters:

```text
response_format=json|zip
num_pages=1
figure_batch_size=1
text_batch_size=8
include_figures=true|false
include_text_reactions=true|false
include_text_molecules=true|false
molscribe=true|false
ocr=true|false
split_large_figures=true|false
deduplicate_figures=true|false
include_figure_overlays=true|false
panel_split_trigger_reactions=0
```

For faster testing, set:

```text
num_pages=1
molscribe=false
ocr=false
include_text_reactions=false
```

For better recall on large multi-panel figures, keep this enabled:

```text
split_large_figures=true
deduplicate_figures=true
include_figure_overlays=true
panel_split_trigger_reactions=0
```

This fallback reruns only large figures that return zero reactions on the first
pass. It selects one non-overlapping split strategy and reports the selected
panels in `metadata.panel_fallbacks`.
De-duplication drops zero-reaction false positives and nested layout crops that
duplicate a larger detected figure.
When `include_figure_overlays=true` and `response_format=zip`, the ZIP includes
`figures/overlays/*.png` images for manual accuracy review.

For richer extraction on H100, try:

```text
molscribe=true
ocr=true
include_text_reactions=true
include_text_molecules=true
```

## Windows PowerShell Examples

Build:

```powershell
docker build -t openchemie-api:test .
```

Run on port `8001`:

```powershell
docker run -d --gpus all --name openchemie-api-test -p 8001:8000 -v openchemie-hf-cache:/app/.cache/huggingface -v openchemie-torch-cache:/app/.cache/torch openchemie-api:test
```

Health:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health"
```

Start:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/start?device=cuda"
```

Test the sample PDF:

```powershell
curl.exe -X POST "http://127.0.0.1:8001/extract-pdf?response_format=zip&num_pages=1&figure_batch_size=1&text_batch_size=8&include_figures=true&include_text_reactions=false&include_text_molecules=true&molscribe=false&ocr=false" -F "pdf=@C:\Users\kavin.kum016\Documents\priya_project\OpenChemIE\example\acs.joc.2c00749.pdf" -o openchemie_first_page.zip
```

Stop models:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/stop"
```

Remove container:

```powershell
docker rm -f openchemie-api-test
```

## Offline Accuracy Audit Script

Use this when you want extraction results plus overlay images and a compact
audit report without starting the FastAPI server:

```bash
python scripts/audit_pdf_reactions.py example/acs.joc.2c00749.pdf --output-dir reaction_audit --output-zip reaction_audit.zip --device cuda
```

The script uses the same high-recall figure workflow:

```text
split_large_figures=true
deduplicate_figures=true
panel_split_trigger_reactions=0
include overlays=true
```

It writes:

```text
reaction_audit/figures_result.json
reaction_audit/reaction_audit_report.json
reaction_audit/overlays/*.png
reaction_audit.zip
```

The report includes page-level reaction counts, panel fallback details,
deduplication drops, and CUDA device information.

## Tested Result In This Workspace

The Docker API was tested with:

```text
C:\Users\kavin.kum016\Documents\priya_project\OpenChemIE\example\acs.joc.2c00749.pdf
```

Local Docker GPU test:

```text
GPU: NVIDIA RTX A500 Laptop GPU
Torch: 2.1.2+cu121
CUDA available: true
```

First-page combined ZIP output:

```text
example/gpu_test_output/docker_combined_first_page.zip
example/gpu_test_output/docker_combined_first_page/
```

Observed counts:

```text
figures_processed: 1
figure reactions: 6
text molecule labels: 1
```
