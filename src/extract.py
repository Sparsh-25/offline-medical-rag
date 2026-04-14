"""
extract.py — Docling-based PDF extraction pipeline
====================================================
Outputs per document (in data/extracted/<doc_id>/):
    <doc_id>.md            ← structured markdown (headings, tables, figure placeholders)
    figures/               ← figure PNGs for captioning
    extraction_log.json    ← page/figure/table counts + figure index
    <doc_id>.json          ← lossless docling dict (only if config.extraction.save_docling_json)

doc_id alignment:
    Uses the same DOC_ID_MAP as meta_builder.py so that extraction output
    and meta.json always land in the same folder.

Idempotency:
    If extraction_log.json already exists for a doc_id, that document is skipped.
    Delete the folder (or just extraction_log.json) to force a re-run.
"""

import json
from pathlib import Path

import yaml
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# ── Config — always resolve relative to this file, not CWD ──────────────────
_ROOT = Path(__file__).parent.parent
cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text())

RAW_DIR       = _ROOT / cfg["paths"]["raw"]
EXTRACTED_DIR = _ROOT / cfg["paths"]["extracted"]
EXTRACTION_CFG = cfg.get("extraction", {})

# ── Use the same doc_id map as meta_builder so folders align ─────────────────
# Maps PDF filename stem → clean doc_id (same as meta_builder.py DOC_ID_MAP)
DOC_ID_MAP: dict[str, str] = {
    "breast guideline": "nccn_breast_v2_2026",
    "wolff-et-al-2023-human-epidermal-growth-factor-receptor-2-testing-in-breast-cancer-asco-college-of-american":
        "asco_her2_testing_guideline_2023",
    "41598_2025_Article_14012":  "sci_reports_nact_hr_pos_2025",
    "fmolb-09-783494":           "frontiers_molbiosci_multiomics_2022",
    "fpubh-11-1049947":          "frontiers_pubhealth_tdxd_cea_2023",
    "s10549-017-4518-8":         "monaleesa2_subanalysis_2018",
}


def _stem_to_doc_id(stem: str) -> str:
    """Return clean doc_id for a PDF stem, falling back to a sanitised stem."""
    if stem in DOC_ID_MAP:
        return DOC_ID_MAP[stem]
    # Sanitise: replace non-alphanumeric runs with underscores, lowercase, truncate
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()[:60]


# ── Converter ────────────────────────────────────────────────────────────────

def build_converter() -> DocumentConverter:
    """
    Build the Docling converter once and reuse across all documents.
    Models load into memory on first call (~30s on CPU, then instant).
    """
    pipeline_opts = PdfPipelineOptions(
        do_ocr=False,                    # digital PDFs only — OCR adds ~5x time
        do_table_structure=True,         # TableFormer — critical for oncology tables
        generate_picture_images=True,    # saves figures as PIL images
        images_scale=2.0,                # 2× resolution for downstream captioning
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)},
    )


# ── Single document extraction ───────────────────────────────────────────────

def extract_document(
    pdf_path: Path,
    converter: DocumentConverter,
    force: bool = False,
) -> dict:
    """
    Extract one PDF. Returns a summary dict.

    Output layout:
        data/extracted/<doc_id>/
            <doc_id>.md              ← structured markdown
            figures/
                figure-000-page-001.png
                ...
            extraction_log.json      ← inventory of what was found
            <doc_id>.json            ← lossless docling dict (optional, config-gated)
    """
    doc_id  = _stem_to_doc_id(pdf_path.stem)
    doc_out = EXTRACTED_DIR / doc_id
    fig_dir = doc_out / "figures"
    log_path = doc_out / "extraction_log.json"

    print(f"\n{'='*55}")
    print(f"PDF   : {pdf_path.name}")
    print(f"doc_id: {doc_id}")

    # ── Idempotency check ─────────────────────────────────────────
    if log_path.exists() and not force:
        existing = json.loads(log_path.read_text())
        print(f"  [✓] Already extracted ({existing.get('figures_found', 0)} figures,"
              f" {existing.get('tables_found', 0)} tables). Skipping.")
        return existing

    doc_out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ── Docling conversion ────────────────────────────────────────
    print("  Converting...")
    result = converter.convert(str(pdf_path))
    doc    = result.document

    # ── 1. Markdown ───────────────────────────────────────────────
    md_text = doc.export_to_markdown(
        image_mode="placeholder",   # writes <!-- image --> placeholders for caption.py
    )
    md_path = doc_out / f"{doc_id}.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"  Markdown → {md_path.name}  ({len(md_text):,} chars)")

    # ── 2. Lossless JSON (optional — debug only) ──────────────────
    if EXTRACTION_CFG.get("save_docling_json", False):
        json_path = doc_out / f"{doc_id}.json"
        json_path.write_text(
            json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Docling JSON → {json_path.name}")

    # ── 3. Figures ────────────────────────────────────────────────
    figures_saved = []
    for pic_idx, picture in enumerate(doc.pictures):
        try:
            img = picture.get_image(doc)
            if img is None:
                continue

            page_no = picture.prov[0].page_no if picture.prov else 0
            filename = f"figure-{pic_idx:03d}-page-{page_no:03d}.png"
            img.save(fig_dir / filename, format="PNG")

            figures_saved.append({
                "index":        pic_idx,
                "page":         page_no,
                "filename":     filename,
                "self_ref":     picture.self_ref,   # docling internal ID in MD placeholder
                "caption_text": None,               # filled by caption.py
            })
            print(f"  Figure {pic_idx:03d} (p{page_no}) → {filename}")

        except Exception as e:
            print(f"  WARNING: figure {pic_idx} skipped — {e}")

    # ── 4. Extraction log ─────────────────────────────────────────
    num_pages = doc.num_pages() if hasattr(doc, "num_pages") else None
    table_count = len(list(doc.tables))

    log = {
        "doc_id":        doc_id,
        "pdf":           str(pdf_path.relative_to(_ROOT)),   # relative path — portable
        "pages":         num_pages,
        "figures_found": len(figures_saved),
        "tables_found":  table_count,
        "figures":       figures_saved,
        "status":        "success",
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"  Pages: {num_pages} | Tables: {table_count} | Figures: {len(figures_saved)}")
    print(f"  Output → {doc_out.relative_to(_ROOT)}/")

    return log


# ── Batch extraction ─────────────────────────────────────────────────────────

def extract_all(force: bool = False):
    """Extract all PDFs in data/raw/. Skips already-extracted docs unless force=True."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}")
        return

    print(f"Found {len(pdfs)} PDFs in {RAW_DIR.relative_to(_ROOT)}/")
    print("Loading Docling models (first run: ~30s on CPU)...")

    converter = build_converter()   # load once, reuse across all docs

    all_logs = []
    for pdf_path in pdfs:
        try:
            log = extract_document(pdf_path, converter, force=force)
            all_logs.append(log)
        except Exception as e:
            print(f"  ERROR: {pdf_path.name} — {e}")
            all_logs.append({"doc_id": _stem_to_doc_id(pdf_path.stem),
                             "pdf": str(pdf_path.relative_to(_ROOT)),
                             "status": "failed", "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    success = [l for l in all_logs if l.get("status") == "success"]
    failed  = [l for l in all_logs if l.get("status") != "success"]
    print(f"Done: {len(success)} success, {len(failed)} failed\n")
    for log in all_logs:
        if log.get("status") == "success":
            print(f"  ✓ {log['doc_id']:<40} {log['figures_found']}fig {log['tables_found']}tbl")
        else:
            print(f"  ✗ {log['doc_id']:<40} FAILED: {log.get('error', '')[:50]}")

    # Master log — written to extracted/ root
    (EXTRACTED_DIR / "_extraction_summary.json").write_text(
        json.dumps(all_logs, indent=2), encoding="utf-8"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Docling PDF extraction pipeline")
    p.add_argument("--pdf",   help="Extract a single PDF (path)")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if extraction_log.json already exists")
    args = p.parse_args()

    converter = build_converter()
    if args.pdf:
        extract_document(Path(args.pdf), converter, force=args.force)
    else:
        extract_all(force=args.force)