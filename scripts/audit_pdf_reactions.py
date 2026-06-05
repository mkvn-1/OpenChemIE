from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import api.figure_pipeline_api as api
from api.figure_pipeline_api import _download_layout_checkpoint, _extract_from_pdf, _without_internal_files
from openchemie import OpenChemIE


def _reaction_counts_by_page(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        page = item.get("page")
        if page is None:
            page_label = "unknown"
        else:
            page_label = str(page + 1)
        counts[page_label] = counts.get(page_label, 0) + len(item.get("reactions", []))
    return counts


def _write_zip(output_zip: Path, payload: dict[str, Any]) -> None:
    overlay_files = payload.get("_overlay_files", [])
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("figures/result.json", json.dumps(_without_internal_files(payload), indent=2, ensure_ascii=False))
        for overlay in overlay_files:
            archive.writestr(f"figures/overlays/{overlay['name']}", overlay["content"])


def _extract_figure_caption_pages(pdf: Path) -> dict[str, list[str]]:
    pdftotext = shutil.which("pdftotext")
    if sys.platform.startswith("win"):
        try:
            candidates = subprocess.run(
                ["where.exe", "pdftotext"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.splitlines()
            pdftotext = next((item for item in candidates if Path(item).suffix.lower() == ".exe"), pdftotext)
        except (OSError, subprocess.CalledProcessError):
            pass
    pages: list[str] = []
    if pdftotext and Path(pdftotext).suffix.lower() in {"", ".exe"}:
        try:
            completed = subprocess.run(
                [pdftotext, "-layout", str(pdf), "-"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            pages = completed.stdout.split("\f")
        except (OSError, subprocess.CalledProcessError):
            pages = []
    if not pages:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(pdf))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            return {}
    figure_pages: dict[str, list[str]] = {}
    figure_pattern = re.compile(r"(?:^|\s{2,})Figure\s+(\d+)\.")
    for page_index, page_text in enumerate(pages, start=1):
        captions = []
        for line in page_text.splitlines():
            if "Figure " not in line:
                continue
            matches = figure_pattern.findall(line)
            if matches:
                captions.extend(f"Figure {match}" for match in matches)
        if captions:
            figure_pages[str(page_index)] = sorted(set(captions), key=lambda item: int(item.split()[1]))
    return figure_pages


def _write_overlay_contact_sheet(output_dir: Path) -> Path | None:
    overlay_dir = output_dir / "overlays"
    overlay_paths = sorted(path for path in overlay_dir.glob("*.png") if path.name.startswith("overlay_"))
    if not overlay_paths:
        return None
    thumbs = []
    for path in overlay_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 360))
        canvas = Image.new("RGB", (390, 410), "white")
        canvas.paste(image, ((390 - image.width) // 2, 35))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), path.stem, fill="black")
        thumbs.append(canvas)
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 390, rows * 410), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 390, (index // columns) * 410))
    sheet_path = output_dir / "overlay_contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def _write_report(output_dir: Path, pdf: Path, payload: dict[str, Any], started: float) -> tuple[Path, dict[str, Any]]:
    metadata = payload["metadata"]
    results = payload["results"]
    figure_caption_pages = _extract_figure_caption_pages(pdf)
    extracted_pages = {str(item["page"] + 1) for item in results if item.get("page") is not None}
    expected_numbered_pages = {
        page
        for page, captions in figure_caption_pages.items()
        if any(caption.startswith("Figure ") for caption in captions)
    }
    # The text extractor also sees prose references. Keep this as a coverage
    # warning, not a hard assertion of figure-page ground truth.
    missing_caption_pages = sorted(expected_numbered_pages - extracted_pages, key=int)
    overlay_contact_sheet = output_dir / "overlay_contact_sheet.png"
    report = {
        "pdf": str(pdf.resolve()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": metadata.get("model", {}).get("device") or str(api._model.device if api._model else "unknown"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "figures_processed": metadata["figures_processed"],
        "total_reactions": metadata["total_reactions"],
        "reaction_counts_by_page": _reaction_counts_by_page(results),
        "pages_with_figures_one_based": sorted({item["page"] + 1 for item in results if item.get("page") is not None}),
        "figure_caption_pages_from_text": figure_caption_pages,
        "caption_pages_without_extractions": missing_caption_pages,
        "split_large_figures": metadata.get("split_large_figures"),
        "deduplicate_figures": metadata.get("deduplicate_figures"),
        "panel_fallbacks_applied": metadata.get("panel_fallbacks_applied"),
        "panel_fallbacks": metadata.get("panel_fallbacks"),
        "dropped_zero_reaction_figures": metadata.get("dropped_zero_reaction_figures"),
        "dropped_duplicate_figures": metadata.get("dropped_duplicate_figures"),
        "deduplication_drops": metadata.get("deduplication_drops"),
        "overlay_count": metadata.get("overlay_count"),
        "overlay_directory": str(output_dir / "overlays"),
        "overlay_contact_sheet": str(overlay_contact_sheet) if overlay_contact_sheet.exists() else None,
        "caveat": "This is an extraction and visual-audit report. Exact all-reaction recall still requires manually labeled ground truth.",
    }
    report_path = output_dir / "reaction_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path, report


def _write_overlays(output_dir: Path, payload: dict[str, Any]) -> None:
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for overlay in payload.get("_overlay_files", []):
        path = overlay_dir / overlay["name"]
        path.write_bytes(overlay["content"])
        summary.append(
            {
                "name": overlay["name"],
                "path": str(path),
                "page": overlay.get("page"),
                "source_figure_index": overlay.get("source_figure_index"),
                "source_panel": overlay.get("source_panel"),
                "reaction_count": overlay.get("reaction_count"),
            }
        )
    (overlay_dir / "overlay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-recall OpenChemIE figure extraction and write audit artifacts.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reaction_audit_output"))
    parser.add_argument("--output-zip", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-pages", type=int, default=None)
    parser.add_argument("--molscribe", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--no-split-large-figures", action="store_true")
    parser.add_argument("--no-deduplicate-figures", action="store_true")
    parser.add_argument("--panel-split-trigger-reactions", type=int, default=0)
    parser.add_argument("--min-panel-split-width", type=int, default=900)
    parser.add_argument("--min-panel-split-height", type=int, default=900)
    parser.add_argument("--fail-on-caption-miss", action="store_true", help="Exit nonzero if any detected caption page has no extraction result.")
    parser.add_argument("--min-total-reactions", type=int, default=None, help="Exit nonzero if total extracted reactions is below this threshold.")
    parser.add_argument("--require-panel-fallback", action="store_true", help="Exit nonzero if no large-figure panel fallback was applied.")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    api._model = OpenChemIE(device=args.device)
    api._model_info = {
        "status": "ready",
        "device": str(api._model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    api._model.init_pdfparser(ckpt_path=_download_layout_checkpoint())
    api._model.init_rxnscribe()

    payload = _extract_from_pdf(
        str(args.pdf),
        batch_size=args.batch_size,
        num_pages=args.num_pages,
        molscribe=args.molscribe,
        ocr=args.ocr,
        split_large_figures=not args.no_split_large_figures,
        deduplicate_figures=not args.no_deduplicate_figures,
        include_overlays=True,
        min_panel_split_width=args.min_panel_split_width,
        min_panel_split_height=args.min_panel_split_height,
        panel_split_trigger_reactions=args.panel_split_trigger_reactions,
    )

    result_path = args.output_dir / "figures_result.json"
    result_path.write_text(json.dumps(_without_internal_files(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_overlays(args.output_dir, payload)
    contact_sheet_path = _write_overlay_contact_sheet(args.output_dir)
    report_path, report = _write_report(args.output_dir, args.pdf, payload, started)

    output_zip = args.output_zip or args.output_dir / "reaction_audit_results.zip"
    _write_zip(output_zip, payload)

    summary = {
        "result_json": str(result_path),
        "audit_report": str(report_path),
        "output_zip": str(output_zip),
        "figures_processed": payload["metadata"]["figures_processed"],
        "total_reactions": payload["metadata"]["total_reactions"],
        "panel_fallbacks_applied": payload["metadata"]["panel_fallbacks_applied"],
        "dropped_zero_reaction_figures": payload["metadata"]["dropped_zero_reaction_figures"],
        "dropped_duplicate_figures": payload["metadata"]["dropped_duplicate_figures"],
        "overlay_count": payload["metadata"]["overlay_count"],
        "overlay_contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
    }
    print(json.dumps(summary, indent=2))

    failures = []
    if args.fail_on_caption_miss and report["caption_pages_without_extractions"]:
        failures.append(
            "caption pages without extraction: "
            + ", ".join(report["caption_pages_without_extractions"])
        )
    if args.min_total_reactions is not None and report["total_reactions"] < args.min_total_reactions:
        failures.append(
            f"total reactions {report['total_reactions']} below required minimum {args.min_total_reactions}"
        )
    if args.require_panel_fallback and report["panel_fallbacks_applied"] < 1:
        failures.append("required panel fallback was not applied")
    if failures:
        raise SystemExit("Audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
