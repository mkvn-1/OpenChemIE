from __future__ import annotations

import io
import gc
import json
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download
from starlette.concurrency import run_in_threadpool

from openchemie import OpenChemIE


app = FastAPI(
    title="OpenChemIE Figure Pipeline API",
    version="0.1.0",
    description="GPU-backed figure reaction extraction for chemistry PDFs.",
)

_state_lock = threading.Lock()
_model: OpenChemIE | None = None
_model_info: dict[str, Any] = {}


def _device_name(device: str) -> str | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        index = 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        return torch.cuda.get_device_name(index)
    return None


def _download_layout_checkpoint() -> str:
    # The DropBox URL bundled in older layoutparser releases is no longer a
    # reliable source. This Hugging Face mirror contains the same PubLayNet D1
    # EfficientDet weights.
    return hf_hub_download(
        repo_id="layoutparser/efficientdet",
        filename="PubLayNet/tf_efficientdet_d1/publaynet-tf_efficientdet_d1.pth.tar",
    )


def _load_models(
    *,
    device: str,
    preload_molscribe: bool,
    preload_moldet: bool,
    preload_coref: bool,
    preload_text_reactions: bool,
    preload_text_molecules: bool,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    started = time.perf_counter()
    model = OpenChemIE(device=device)

    layout_checkpoint = _download_layout_checkpoint()
    model.init_pdfparser(ckpt_path=layout_checkpoint)
    model.init_rxnscribe()

    loaded = ["pdfparser", "rxnscribe"]
    if preload_molscribe:
        model.init_molscribe()
        loaded.append("molscribe")
    if preload_moldet:
        model.init_moldet()
        loaded.append("moldet")
    if preload_coref:
        model.init_coref()
        loaded.append("coref")
    if preload_text_reactions:
        model.init_chemrxnextractor()
        loaded.append("chemrxnextractor")
    if preload_text_molecules:
        model.init_chemner()
        if "chemrxnextractor" not in loaded:
            model.init_chemrxnextractor()
            loaded.append("chemrxnextractor")
        loaded.append("chemner")

    info = {
        "status": "ready",
        "device": str(model.device),
        "device_name": _device_name(str(model.device)),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "loaded": loaded,
        "layout_checkpoint": layout_checkpoint,
        "load_seconds": round(time.perf_counter() - started, 3),
    }

    global _model, _model_info
    with _state_lock:
        _model = model
        _model_info = info

    return info


def _require_model() -> OpenChemIE:
    with _state_lock:
        model = _model
    if model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Call POST /start first.")
    return model


def _stop_models() -> dict[str, Any]:
    global _model, _model_info
    with _state_lock:
        was_loaded = _model is not None
        previous = dict(_model_info)
        model = _model
        if model is not None:
            for method_name in (
                "init_molscribe",
                "init_rxnscribe",
                "init_pdfparser",
                "init_moldet",
                "init_coref",
                "init_chemrxnextractor",
                "init_chemner",
            ):
                method = getattr(model, method_name, None)
                cache_clear = getattr(method, "cache_clear", None)
                if cache_clear is not None:
                    cache_clear()
            for attr_name in (
                "_molscribe",
                "_rxnscribe",
                "_pdfparser",
                "_moldet",
                "_coref",
                "_chemrxnextractor",
                "_chemner",
            ):
                setattr(model, attr_name, None)
        _model = None
        _model_info = {}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "status": "stopped",
        "was_loaded": was_loaded,
        "previous_model": previous,
        "cuda_available": torch.cuda.is_available(),
        "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
    }


def _component_status(model: OpenChemIE | None) -> dict[str, bool]:
    if model is None:
        return {
            "pdfparser": False,
            "rxnscribe": False,
            "molscribe": False,
            "moldet": False,
            "coref": False,
            "chemrxnextractor": False,
            "chemner": False,
        }
    return {
        "pdfparser": model._pdfparser is not None,
        "rxnscribe": model._rxnscribe is not None,
        "molscribe": model._molscribe is not None,
        "moldet": model._moldet is not None,
        "coref": model._coref is not None,
        "chemrxnextractor": model._chemrxnextractor is not None,
        "chemner": model._chemner is not None,
    }


def _cuda_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": 0,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if key not in {"image", "figure"}
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    reaction_counts = [len(item.get("reactions", [])) for item in results]
    pages = sorted({item.get("page") for item in results if item.get("page") is not None})
    return {
        "figures_processed": len(results),
        "pages_with_figures": pages,
        "reaction_counts_by_figure": reaction_counts,
        "total_reactions": sum(reaction_counts),
    }


def _extract_from_pdf(
    pdf_path: str,
    *,
    batch_size: int,
    num_pages: int | None,
    molscribe: bool,
    ocr: bool,
) -> dict[str, Any]:
    model = _require_model()
    started = time.perf_counter()
    results = model.extract_reactions_from_figures_in_pdf(
        pdf_path,
        batch_size=batch_size,
        num_pages=num_pages,
        molscribe=molscribe,
        ocr=ocr,
    )
    clean_results = _jsonable(results)
    return {
        "metadata": {
            "method": "OpenChemIE.extract_reactions_from_figures_in_pdf",
            "batch_size": batch_size,
            "num_pages": num_pages,
            "molscribe": molscribe,
            "ocr": ocr,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": _model_info,
            **_summarize(clean_results),
        },
        "results": clean_results,
    }


def _extract_text_reactions(pdf_path: str, *, num_pages: int | None) -> dict[str, Any]:
    model = _require_model()
    started = time.perf_counter()
    results = model.extract_reactions_from_text_in_pdf(pdf_path, num_pages=num_pages)
    clean_results = _jsonable(results)
    reaction_sentence_counts = [len(page.get("reactions", [])) for page in clean_results]
    return {
        "metadata": {
            "method": "OpenChemIE.extract_reactions_from_text_in_pdf",
            "num_pages": num_pages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": _model_info,
            "page_results": len(clean_results),
            "reaction_sentence_counts_by_page": reaction_sentence_counts,
            "total_reaction_sentences": sum(reaction_sentence_counts),
        },
        "results": clean_results,
    }


def _extract_text_molecules(pdf_path: str, *, batch_size: int, num_pages: int | None) -> dict[str, Any]:
    model = _require_model()
    started = time.perf_counter()
    results = model.extract_molecules_from_text_in_pdf(
        pdf_path,
        batch_size=batch_size,
        num_pages=num_pages,
    )
    clean_results = _jsonable(results)
    paragraph_counts = [len(page.get("molecules", [])) for page in clean_results]
    label_counts = [
        sum(len(paragraph.get("labels", [])) for paragraph in page.get("molecules", []))
        for page in clean_results
    ]
    return {
        "metadata": {
            "method": "OpenChemIE.extract_molecules_from_text_in_pdf",
            "batch_size": batch_size,
            "num_pages": num_pages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": _model_info,
            "page_results": len(clean_results),
            "paragraph_counts_by_page": paragraph_counts,
            "label_counts_by_page": label_counts,
            "total_labels": sum(label_counts),
        },
        "results": clean_results,
    }


def _extract_combined_pdf(
    pdf_path: str,
    *,
    figure_batch_size: int,
    text_batch_size: int,
    num_pages: int | None,
    include_figures: bool,
    include_text_reactions: bool,
    include_text_molecules: bool,
    molscribe: bool,
    ocr: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "metadata": {
            "method": "combined OpenChemIE PDF extraction",
            "num_pages": num_pages,
            "include_figures": include_figures,
            "include_text_reactions": include_text_reactions,
            "include_text_molecules": include_text_molecules,
            "model": _model_info,
        }
    }

    if include_figures:
        payload["figures"] = _extract_from_pdf(
            pdf_path,
            batch_size=figure_batch_size,
            num_pages=num_pages,
            molscribe=molscribe,
            ocr=ocr,
        )
    if include_text_reactions:
        payload["text_reactions"] = _extract_text_reactions(pdf_path, num_pages=num_pages)
    if include_text_molecules:
        payload["text_molecules"] = _extract_text_molecules(
            pdf_path,
            batch_size=text_batch_size,
            num_pages=num_pages,
        )

    payload["metadata"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return payload


@app.post("/start")
async def start(
    device: str = Query("cuda", description="Use cuda, cuda:0, cuda:1, or cpu."),
    preload_molscribe: bool = Query(False, description="Load MolScribe at startup for SMILES/molfile output."),
    preload_moldet: bool = Query(False, description="Load molecule detector at startup."),
    preload_coref: bool = Query(False, description="Load molecule coreference model at startup."),
    preload_text_reactions: bool = Query(False, description="Load ChemRxnExtractor text reaction model at startup."),
    preload_text_molecules: bool = Query(False, description="Load ChemNER text molecule model at startup."),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            _load_models,
            device=device,
            preload_molscribe=preload_molscribe,
            preload_moldet=preload_moldet,
            preload_coref=preload_coref,
            preload_text_reactions=preload_text_reactions,
            preload_text_molecules=preload_text_molecules,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    with _state_lock:
        model = _model
        loaded = model is not None
        model_info = dict(_model_info)
    components = _component_status(model)
    return {
        "loaded": loaded,
        "status": "ready" if loaded else "stopped",
        "model": model_info,
        "components": components,
        "loaded_components": [name for name, is_loaded in components.items() if is_loaded],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_memory": _cuda_memory(),
    }


@app.post("/stop")
async def stop() -> dict[str, Any]:
    return await run_in_threadpool(_stop_models)


@app.post("/extract-figures")
async def extract_figures(
    pdf: UploadFile = File(...),
    response_format: Literal["json", "zip"] = Query("json", description="Return JSON directly or a ZIP containing result.json."),
    batch_size: int = Query(1, ge=1),
    num_pages: int | None = Query(None, ge=1, description="Limit to the first N pages."),
    molscribe: bool = Query(False, description="Return SMILES/molfile for molecule images. Slower and uses more VRAM."),
    ocr: bool = Query(False, description="OCR reaction condition text. Slower and uses more VRAM."),
) -> Response:
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a PDF file.")

    suffix = Path(pdf.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        temp.write(await pdf.read())

    try:
        payload = await run_in_threadpool(
            _extract_from_pdf,
            str(temp_path),
            batch_size=batch_size,
            num_pages=num_pages,
            molscribe=molscribe,
            ocr=ocr,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if response_format == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("result.json", json.dumps(payload, indent=2, ensure_ascii=False))
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=openchemie_figure_results.zip"},
        )

    return JSONResponse(payload)


@app.post("/extract-pdf")
async def extract_pdf(
    pdf: UploadFile = File(...),
    response_format: Literal["json", "zip"] = Query("zip", description="Return JSON directly or a ZIP with separate figures/text folders."),
    figure_batch_size: int = Query(1, ge=1),
    text_batch_size: int = Query(8, ge=1),
    num_pages: int | None = Query(None, ge=1, description="Limit to the first N pages."),
    include_figures: bool = Query(True),
    include_text_reactions: bool = Query(True),
    include_text_molecules: bool = Query(True),
    molscribe: bool = Query(False, description="Return SMILES/molfile for figure molecules. Slower and uses more VRAM."),
    ocr: bool = Query(False, description="OCR figure reaction condition text. Slower and uses more VRAM."),
) -> Response:
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a PDF file.")
    if not (include_figures or include_text_reactions or include_text_molecules):
        raise HTTPException(status_code=400, detail="At least one extraction pipeline must be enabled.")

    suffix = Path(pdf.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        temp.write(await pdf.read())

    try:
        payload = await run_in_threadpool(
            _extract_combined_pdf,
            str(temp_path),
            figure_batch_size=figure_batch_size,
            text_batch_size=text_batch_size,
            num_pages=num_pages,
            include_figures=include_figures,
            include_text_reactions=include_text_reactions,
            include_text_molecules=include_text_molecules,
            molscribe=molscribe,
            ocr=ocr,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if response_format == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
            if "figures" in payload:
                archive.writestr("figures/result.json", json.dumps(payload["figures"], indent=2, ensure_ascii=False))
            if "text_reactions" in payload:
                archive.writestr("text/reactions.json", json.dumps(payload["text_reactions"], indent=2, ensure_ascii=False))
            if "text_molecules" in payload:
                archive.writestr("text/molecules.json", json.dumps(payload["text_molecules"], indent=2, ensure_ascii=False))
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=openchemie_pdf_results.zip"},
        )

    return JSONResponse(payload)
