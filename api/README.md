# OpenChemIE Figure Pipeline API

This FastAPI app keeps OpenChemIE's figure models loaded on GPU and exposes PDF extraction over HTTP.

## Install

Use a GPU environment with CUDA PyTorch and the OpenChemIE dependencies installed. Add the API dependencies:

```bash
pip install fastapi "uvicorn[standard]" python-multipart
```

Poppler must also be installed and available on `PATH` for `pdf2image`.

## Run

```bash
uvicorn api.figure_pipeline_api:app --host 0.0.0.0 --port 8000
```

## Load Models

Load the layout detector and RxnScribe onto GPU:

```bash
curl -X POST "http://localhost:8000/start?device=cuda"
```

For richer molecule output, preload MolScribe too:

```bash
curl -X POST "http://localhost:8000/start?device=cuda&preload_molscribe=true"
```

Unload models and release CUDA cache:

```bash
curl -X POST "http://localhost:8000/stop"
```

## Extract From PDF

Fastest figure reaction extraction:

```bash
curl -X POST "http://localhost:8000/extract-figures?batch_size=1&molscribe=false&ocr=false" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o result.json
```

Return a ZIP containing `result.json`:

```bash
curl -X POST "http://localhost:8000/extract-figures?response_format=zip&batch_size=1" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o result.zip
```

## Combined Figure + Text Extraction

Use `/extract-pdf` when you want both figure results and text results in one ZIP. The ZIP is structured as:

```text
metadata.json
figures/result.json
text/reactions.json
text/molecules.json
```

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&figure_batch_size=1&text_batch_size=8" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o openchemie_results.zip
```

For a quick smoke test, limit the work to the first page:

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&num_pages=1&figure_batch_size=1&text_batch_size=8" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o openchemie_first_page.zip
```

You can disable expensive parts independently:

```bash
curl -X POST "http://localhost:8000/extract-pdf?include_text_molecules=false&molscribe=false&ocr=false" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o openchemie_results.zip
```

Large multi-panel figures can be split and retried automatically when the
first pass finds too few reactions:

```bash
curl -X POST "http://localhost:8000/extract-pdf?response_format=zip&split_large_figures=true&deduplicate_figures=true&panel_split_trigger_reactions=0&include_figure_overlays=true" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o openchemie_results.zip
```

The fallback is enabled by default. It is useful for large figures where one
detected crop contains several panels and RxnScribe misses the full crop.
De-duplication is also enabled by default to drop zero-reaction false positives
and nested layout crops that duplicate a larger detected figure.
When `include_figure_overlays=true` and `response_format=zip`, the ZIP includes
`figures/overlays/*.png` images for manual accuracy review.

Fuller but slower output:

```bash
curl -X POST "http://localhost:8000/extract-figures?batch_size=1&molscribe=true&ocr=true" \
  -F "pdf=@example/acs.joc.2c00749.pdf" \
  -o result.json
```

On an H100, `molscribe=true` and `ocr=true` are much more practical than on a small laptop GPU, but start with `batch_size=1` and increase only after observing VRAM use.
