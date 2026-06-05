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
from PIL import ImageDraw
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


def _without_internal_files(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _without_internal_files(value)
            for key, value in payload.items()
            if key != "_overlay_files"
        }
    if isinstance(payload, list):
        return [_without_internal_files(item) for item in payload]
    return payload


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    reaction_counts = [len(item.get("reactions", [])) for item in results]
    pages = sorted({item.get("page") for item in results if item.get("page") is not None})
    return {
        "figures_processed": len(results),
        "pages_with_figures": pages,
        "reaction_counts_by_figure": reaction_counts,
        "total_reactions": sum(reaction_counts),
    }


def _bbox_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip("[]()")
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            return None
        return [float(part) for part in parts]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(part) for part in value]
    return None


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_intersection(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    intersection = _bbox_intersection(a, b)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union else 0.0


def _bbox_containment(inner: list[float], outer: list[float]) -> float:
    inner_area = _bbox_area(inner)
    return _bbox_intersection(inner, outer) / inner_area if inner_area else 0.0


def _panel_bbox_on_page(source_bbox: list[float] | None, source_size: tuple[int, int], panel_bbox: tuple[int, int, int, int]) -> list[float] | None:
    if source_bbox is None:
        return None
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        return None
    page_width = source_bbox[2] - source_bbox[0]
    page_height = source_bbox[3] - source_bbox[1]
    x1, y1, x2, y2 = panel_bbox
    return [
        source_bbox[0] + page_width * (x1 / source_width),
        source_bbox[1] + page_height * (y1 / source_height),
        source_bbox[0] + page_width * (x2 / source_width),
        source_bbox[1] + page_height * (y2 / source_height),
    ]


def _deduplicate_figure_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [result for result in results if len(result.get("reactions", [])) > 0]
    dropped: list[dict[str, Any]] = []

    grouped: dict[Any, list[dict[str, Any]]] = {}
    for result in candidates:
        grouped.setdefault(result.get("page"), []).append(result)

    kept: list[dict[str, Any]] = []
    for page, page_results in grouped.items():
        ordered = sorted(
            page_results,
            key=lambda item: (
                len(item.get("reactions", [])),
                _bbox_area(item.get("source_figure_bbox") or [0, 0, 0, 0]),
            ),
            reverse=True,
        )
        page_kept: list[dict[str, Any]] = []
        for result in ordered:
            bbox = result.get("source_figure_bbox")
            should_drop = False
            for existing in page_kept:
                existing_bbox = existing.get("source_figure_bbox")
                if not bbox or not existing_bbox:
                    continue
                iou = _bbox_iou(bbox, existing_bbox)
                contained = _bbox_containment(bbox, existing_bbox)
                existing_reactions = len(existing.get("reactions", []))
                result_reactions = len(result.get("reactions", []))
                if iou >= 0.80 or (contained >= 0.88 and existing_reactions >= result_reactions):
                    should_drop = True
                    dropped.append(
                        {
                            "page": page,
                            "source_figure_index": result.get("source_figure_index"),
                            "reaction_count": result_reactions,
                            "reason": "overlapping_or_contained_duplicate",
                            "overlap_with_source_figure_index": existing.get("source_figure_index"),
                            "iou": round(iou, 4),
                            "containment": round(contained, 4),
                        }
                    )
                    break
            if not should_drop:
                page_kept.append(result)
        kept.extend(sorted(page_kept, key=lambda item: item.get("source_figure_index", 0)))

    zero_reaction_drops = [
        {
            "page": result.get("page"),
            "source_figure_index": result.get("source_figure_index"),
            "reason": "zero_reactions",
        }
        for result in results
        if len(result.get("reactions", [])) == 0
    ]
    metadata = {
        "deduplicate_figures": True,
        "dropped_zero_reaction_figures": len(zero_reaction_drops),
        "dropped_duplicate_figures": len(dropped),
        "deduplication_drops": zero_reaction_drops + dropped,
    }
    return kept, metadata


def _split_large_figure(image: Any) -> list[dict[str, Any]]:
    width, height = image.size
    midpoint_x = width // 2
    midpoint_y = height // 2
    strategies = [
        (
            "vertical_halves",
            [
                ("left", (0, 0, midpoint_x, height)),
                ("right", (midpoint_x, 0, width, height)),
            ],
        ),
        (
            "horizontal_halves",
            [
                ("top", (0, 0, width, midpoint_y)),
                ("bottom", (0, midpoint_y, width, height)),
            ],
        ),
        (
            "quadrants",
            [
                ("top_left", (0, 0, midpoint_x, midpoint_y)),
                ("top_right", (midpoint_x, 0, width, midpoint_y)),
                ("bottom_left", (0, midpoint_y, midpoint_x, height)),
                ("bottom_right", (midpoint_x, midpoint_y, width, height)),
            ],
        ),
    ]
    candidates = []
    for strategy, boxes in strategies:
        panels = []
        for name, box in boxes:
            x1, y1, x2, y2 = box
            if x2 - x1 < 180 or y2 - y1 < 180:
                continue
            panels.append(
                {
                    "strategy": strategy,
                    "panel": name,
                    "bbox_in_figure": box,
                    "image": image.crop(box),
                }
            )
        if panels:
            candidates.append({"strategy": strategy, "panels": panels})
    return candidates


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, size: tuple[int, int], color: str) -> None:
    bbox = _bbox_to_list(bbox)
    if bbox is None:
        return
    width, height = size
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
    else:
        box = [x1, y1, x2, y2]
    draw.rectangle(box, outline=color, width=3)


def _build_reaction_overlays(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    colors = {
        "reactants": "red",
        "products": "blue",
        "conditions": "orange",
    }
    for output_index, result in enumerate(results, start=1):
        image = result.get("figure")
        if image is None:
            continue
        image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(image)
        reaction_count = len(result.get("reactions", []))
        title = (
            f"out {output_index} page {result.get('page', -1) + 1} "
            f"src {result.get('source_figure_index', output_index)} "
            f"rxns {reaction_count}"
        )
        if result.get("source_panel"):
            title += f" panel {result['source_panel']}"
        draw.rectangle([0, 0, min(image.size[0], 720), 26], fill="white")
        draw.text((6, 5), title, fill="black")
        for reaction_index, reaction in enumerate(result.get("reactions", []), start=1):
            draw.text((8, 26 + (reaction_index - 1) * 18), f"R{reaction_index}", fill="black")
            for role, color in colors.items():
                for item in reaction.get(role, []):
                    _draw_bbox(draw, item.get("bbox"), image.size, color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        overlays.append(
            {
                "name": (
                    f"overlay_{output_index:02d}_page_{result.get('page', -1) + 1:02d}_"
                    f"src_{result.get('source_figure_index', output_index):02d}"
                    f"{'_' + result['source_panel'] if result.get('source_panel') else ''}.png"
                ),
                "content": buffer.getvalue(),
                "page": result.get("page"),
                "source_figure_index": result.get("source_figure_index"),
                "source_panel": result.get("source_panel"),
                "reaction_count": reaction_count,
            }
        )
    return overlays


def _extract_from_pdf_with_panel_fallback(
    model: OpenChemIE,
    pdf_path: str,
    *,
    batch_size: int,
    num_pages: int | None,
    molscribe: bool,
    ocr: bool,
    split_large_figures: bool,
    deduplicate_figures: bool,
    include_overlays: bool,
    min_panel_split_width: int,
    min_panel_split_height: int,
    panel_split_trigger_reactions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    figures = model.extract_figures_from_pdf(pdf_path, num_pages=num_pages, output_bbox=True)
    images = [figure["figure"]["image"] for figure in figures]
    base_results = model.extract_reactions_from_figures(
        images,
        batch_size=batch_size,
        molscribe=molscribe,
        ocr=ocr,
    )

    fallback_records: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []
    for index, (figure, result) in enumerate(zip(figures, base_results), start=1):
        result["page"] = figure["page"]
        source_bbox = _bbox_to_list(figure["figure"].get("bbox"))
        result["source_figure_index"] = index
        result["source_figure_bbox"] = source_bbox
        image = figure["figure"]["image"]
        width, height = image.size
        result["source_figure_size"] = [width, height]
        reaction_count = len(result.get("reactions", []))
        should_split = (
            split_large_figures
            and width >= min_panel_split_width
            and height >= min_panel_split_height
            and reaction_count <= panel_split_trigger_reactions
        )
        if not should_split:
            final_results.append(result)
            continue

        best_strategy: dict[str, Any] | None = None
        for candidate in _split_large_figure(image):
            panels = candidate["panels"]
            panel_results = model.extract_reactions_from_figures(
                [panel["image"] for panel in panels],
                batch_size=batch_size,
                molscribe=molscribe,
                ocr=ocr,
            )
            accepted_panels: list[dict[str, Any]] = []
            for panel, panel_result in zip(panels, panel_results):
                panel_reaction_count = len(panel_result.get("reactions", []))
                if panel_reaction_count <= 0:
                    continue
                panel_result["page"] = figure["page"]
                panel_result["source_figure_index"] = index
                panel_result["source_split_strategy"] = candidate["strategy"]
                panel_result["source_panel"] = panel["panel"]
                panel_result["panel_bbox_in_source_figure"] = panel["bbox_in_figure"]
                panel_result["source_figure_bbox"] = _panel_bbox_on_page(
                    source_bbox,
                    (width, height),
                    panel["bbox_in_figure"],
                )
                panel_result["source_figure_size"] = [
                    panel["bbox_in_figure"][2] - panel["bbox_in_figure"][0],
                    panel["bbox_in_figure"][3] - panel["bbox_in_figure"][1],
                ]
                accepted_panels.append(panel_result)
            candidate_total = sum(len(panel.get("reactions", [])) for panel in accepted_panels)
            if best_strategy is None or candidate_total > best_strategy["total_reactions"]:
                best_strategy = {
                    "strategy": candidate["strategy"],
                    "total_reactions": candidate_total,
                    "accepted_panels": accepted_panels,
                }

        accepted_panels = []
        if best_strategy and best_strategy["total_reactions"] > reaction_count:
            accepted_panels = best_strategy["accepted_panels"]

        if accepted_panels:
            final_results.extend(accepted_panels)
            fallback_records.append(
                {
                    "source_figure_index": index,
                    "page": figure["page"],
                    "source_size": [width, height],
                    "source_reactions": reaction_count,
                    "selected_strategy": best_strategy["strategy"] if best_strategy else None,
                    "accepted_panels": [
                        {
                            "panel": panel["source_panel"],
                            "strategy": panel["source_split_strategy"],
                            "reaction_count": len(panel.get("reactions", [])),
                            "bbox_in_source_figure": panel["panel_bbox_in_source_figure"],
                        }
                        for panel in accepted_panels
                    ],
                }
            )
        else:
            final_results.append(result)

    if deduplicate_figures:
        final_results, deduplication_metadata = _deduplicate_figure_results(final_results)
    else:
        deduplication_metadata = {
            "deduplicate_figures": False,
            "dropped_zero_reaction_figures": 0,
            "dropped_duplicate_figures": 0,
            "deduplication_drops": [],
        }

    metadata = {
        "split_large_figures": split_large_figures,
        "deduplicate_figures": deduplicate_figures,
        "include_overlays": include_overlays,
        "min_panel_split_width": min_panel_split_width,
        "min_panel_split_height": min_panel_split_height,
        "panel_split_trigger_reactions": panel_split_trigger_reactions,
        "panel_fallbacks_applied": len(fallback_records),
        "panel_fallbacks": fallback_records,
        **deduplication_metadata,
    }
    return final_results, metadata


def _extract_from_pdf(
    pdf_path: str,
    *,
    batch_size: int,
    num_pages: int | None,
    molscribe: bool,
    ocr: bool,
    split_large_figures: bool,
    deduplicate_figures: bool,
    include_overlays: bool,
    min_panel_split_width: int,
    min_panel_split_height: int,
    panel_split_trigger_reactions: int,
) -> dict[str, Any]:
    model = _require_model()
    started = time.perf_counter()
    results, fallback_metadata = _extract_from_pdf_with_panel_fallback(
        model,
        pdf_path,
        batch_size=batch_size,
        num_pages=num_pages,
        molscribe=molscribe,
        ocr=ocr,
        split_large_figures=split_large_figures,
        deduplicate_figures=deduplicate_figures,
        include_overlays=include_overlays,
        min_panel_split_width=min_panel_split_width,
        min_panel_split_height=min_panel_split_height,
        panel_split_trigger_reactions=panel_split_trigger_reactions,
    )
    overlays = _build_reaction_overlays(results) if include_overlays else []
    clean_results = _jsonable(results)
    payload = {
        "metadata": {
            "method": "OpenChemIE.extract_reactions_from_figures_in_pdf",
            "batch_size": batch_size,
            "num_pages": num_pages,
            "molscribe": molscribe,
            "ocr": ocr,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": _model_info,
            **fallback_metadata,
            **_summarize(clean_results),
        },
        "results": clean_results,
    }
    if overlays:
        payload["_overlay_files"] = overlays
        payload["metadata"]["overlay_count"] = len(overlays)
    else:
        payload["metadata"]["overlay_count"] = 0
    return payload


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
    split_large_figures: bool,
    deduplicate_figures: bool,
    include_figure_overlays: bool,
    min_panel_split_width: int,
    min_panel_split_height: int,
    panel_split_trigger_reactions: int,
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
            split_large_figures=split_large_figures,
            deduplicate_figures=deduplicate_figures,
            include_overlays=include_figure_overlays,
            min_panel_split_width=min_panel_split_width,
            min_panel_split_height=min_panel_split_height,
            panel_split_trigger_reactions=panel_split_trigger_reactions,
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
    split_large_figures: bool = Query(True, description="Rerun large low-recall figures as smaller panels."),
    deduplicate_figures: bool = Query(True, description="Drop zero-reaction and overlapping duplicate figure crops."),
    include_overlays: bool = Query(False, description="Include detection overlay PNGs in ZIP responses for manual audit."),
    min_panel_split_width: int = Query(900, ge=1),
    min_panel_split_height: int = Query(900, ge=1),
    panel_split_trigger_reactions: int = Query(0, ge=0, description="Split only when a large figure has this many reactions or fewer."),
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
            split_large_figures=split_large_figures,
            deduplicate_figures=deduplicate_figures,
            include_overlays=include_overlays,
            min_panel_split_width=min_panel_split_width,
            min_panel_split_height=min_panel_split_height,
            panel_split_trigger_reactions=panel_split_trigger_reactions,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if response_format == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            overlay_files = payload.get("_overlay_files", [])
            archive.writestr("result.json", json.dumps(_without_internal_files(payload), indent=2, ensure_ascii=False))
            for overlay in overlay_files:
                archive.writestr(f"overlays/{overlay['name']}", overlay["content"])
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=openchemie_figure_results.zip"},
        )

    return JSONResponse(_without_internal_files(payload))


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
    split_large_figures: bool = Query(True, description="Rerun large low-recall figures as smaller panels."),
    deduplicate_figures: bool = Query(True, description="Drop zero-reaction and overlapping duplicate figure crops."),
    include_figure_overlays: bool = Query(False, description="Include figure detection overlay PNGs in ZIP responses for manual audit."),
    min_panel_split_width: int = Query(900, ge=1),
    min_panel_split_height: int = Query(900, ge=1),
    panel_split_trigger_reactions: int = Query(0, ge=0, description="Split only when a large figure has this many reactions or fewer."),
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
            split_large_figures=split_large_figures,
            deduplicate_figures=deduplicate_figures,
            include_figure_overlays=include_figure_overlays,
            min_panel_split_width=min_panel_split_width,
            min_panel_split_height=min_panel_split_height,
            panel_split_trigger_reactions=panel_split_trigger_reactions,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if response_format == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
            if "figures" in payload:
                overlay_files = payload["figures"].get("_overlay_files", [])
                archive.writestr("figures/result.json", json.dumps(_without_internal_files(payload["figures"]), indent=2, ensure_ascii=False))
                for overlay in overlay_files:
                    archive.writestr(f"figures/overlays/{overlay['name']}", overlay["content"])
            if "text_reactions" in payload:
                archive.writestr("text/reactions.json", json.dumps(payload["text_reactions"], indent=2, ensure_ascii=False))
            if "text_molecules" in payload:
                archive.writestr("text/molecules.json", json.dumps(payload["text_molecules"], indent=2, ensure_ascii=False))
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=openchemie_pdf_results.zip"},
        )

    return JSONResponse(_without_internal_files(payload))
