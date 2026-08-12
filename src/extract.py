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
FIG_MIN_W = EXTRACTION_CFG.get("figure_min_width",  250)   # px
FIG_MIN_H = EXTRACTION_CFG.get("figure_min_height", 200)   # px

# ── Known PDF font-encoding glyph corruption ─────────────────────────────────
# Some source PDFs use custom/subset fonts whose cmap tables can't be resolved
# for a handful of symbol glyphs. Confirmed (decisions.md D28) that this exists
# in the PDF's own text layer, before Docling touches it — verified via raw
# pdfium text extraction — so no Docling backend/OCR setting can avoid it.
# Forced full-page OCR *can* recover the correct symbols but was found to
# silently drop table header/row-label cells instead — worse than this fix.
# These codepoints have no legitimate use in English clinical text.
_GLYPH_FIXES: dict[str, str] = {
    "þ": "+",       # þ (thorn) — HIGH confidence: every instance was "+" (IHC 2þ, HR þ, A þ B)
    "¼": "=",       # HIGH confidence: every instance was "=" in a stat (P ¼ 0.003, n ¼ 436) — 140 instances found
    "\x00":   "−",  # NUL — HIGH confidence: always between digits in a difference/CI range
    "\x06":   "±",  # HIGH confidence: single instance, a "day 1 (±3 days)" visit window
    "\x15":   "≥",  # MEDIUM confidence: inferred from paired table-row context, not confirmed against source PDF
    "\x14":   "<",        # MEDIUM confidence: same table, paired with \x15 above
    "\x03":   "",         # stray artifact in an author-affiliation list — no clear intended character, stripped
    "\x13":   "",         # stray artifact in an author-affiliation list — no clear intended character, stripped
}
# NOT fixed (deliberately): U+FFFD, €, and two Braille-pattern characters also
# turned up in this corpus's author/affiliation/reference-citation text,
# standing in for various German/Polish diacritics (é, ä, ü, ń). Left alone —
# each corrupted codepoint maps to a *different* intended letter depending on
# context, so a blind substitution risks being wrong, and the affected text
# (author names, city names, references) carries no clinical/retrieval
# significance for this project. See decisions.md D28.


def _fix_known_glyph_corruption(text: str) -> str:
    """Repair known PDF font-encoding corruption (see decisions.md D28). No-op if none present."""
    for bad, good in _GLYPH_FIXES.items():
        text = text.replace(bad, good)
    return text

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
    "ESMO Clinical Practice Guideline- Metastatic Breast Cancer_2026":
        "esmo_metastatic_bc_guideline_2026",
    "NCCN Guidelines Genetic:Familial High-Risk Assessment- Breast, Ovarian, Pancreatic, and Prostate_2026":
        "nccn_genetic_familial_risk_v3_2026",
    "Neoadjuvant pembrolizumab plus chemotherapy:adjuvant pembrolizumab for early-stage triple-negative breast cancer- quality-of-life results from the randomized KEYNOTE-522 study_2025":
        "keynote522_tnbc_qol_2025",
    "Overall survival in the OlympiA phase III trial of adjuvant olaparib in patients with germline pathogenic variants in BRCA1:2 and high-risk, early breast cancer_2022":
        "olympia_brca_os_2022",
    "destiny06_2025":            "destiny_breast06_2025",
    'pubmed_meta_"CDK4:6 inhibitor_2023':
        "cdk46i_meta_analysis_2023",
}


def _get_doc_id(pdf_path: Path) -> str:
    """Return clean doc_id for a PDF, incorporating an MD5 hash of its contents for true uniqueness."""
    stem = pdf_path.stem
    if stem in DOC_ID_MAP:
        return DOC_ID_MAP[stem]
    import re, hashlib
    clean_stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()[:50]
    content_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()[:8]
    return f"{clean_stem}_{content_hash}"


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
    doc_id  = _get_doc_id(pdf_path)
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
    md_text = _fix_known_glyph_corruption(md_text)
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

    # ── 3. Figures (filtered + deduplicated) ───────────────────
    figures_saved = []
    seen_hashes: set[bytes] = set()   # content-hash dedup within this doc

    for pic_idx, picture in enumerate(doc.pictures):
        try:
            img = picture.get_image(doc)
            if img is None:
                continue

            page_no  = picture.prov[0].page_no if picture.prov else 0

            # ── Size filter: skip logos, icons, thin banners ──
            w, h = img.size
            if w < FIG_MIN_W or h < FIG_MIN_H:
                print(f"  Skip  {pic_idx:03d} ({w}x{h}) — too small")
                figures_saved.append({
                    "index":        pic_idx,
                    "page":         page_no,
                    "filename":     None,
                    "status":       "skipped",
                    "skip_reason":  "too_small",
                    "self_ref":     picture.self_ref,
                })
                continue

            # ── Content dedup: skip identical images (repeated logos) ──
            img_hash = img.tobytes()
            if img_hash in seen_hashes:
                print(f"  Skip  {pic_idx:03d} ({w}x{h}) — duplicate")
                figures_saved.append({
                    "index":        pic_idx,
                    "page":         page_no,
                    "filename":     None,
                    "status":       "skipped",
                    "skip_reason":  "duplicate",
                    "self_ref":     picture.self_ref,
                })
                continue
            seen_hashes.add(img_hash)

            filename = f"figure-{pic_idx:03d}-page-{page_no:03d}.png"
            img.save(fig_dir / filename, format="PNG")

            # Capture Docling's extracted caption as context for the VLM.
            # Docling ≥ 2.x uses picture.captions (List[RefItem]) instead of
            # picture.caption (single object). Resolve each ref to get text.
            docling_context = ""
            for cap_ref in getattr(picture, "captions", []):
                try:
                    resolved = cap_ref.resolve(doc)
                    text = getattr(resolved, "text", None) or ""
                    docling_context += text.strip() + " "
                except Exception:
                    pass
            # Fallback: try legacy API for older Docling installs
            if not docling_context.strip():
                try:
                    legacy = getattr(picture, "caption", None)
                    if legacy:
                        docling_context = getattr(legacy, "text", "") or ""
                except Exception:
                    pass
            docling_context = docling_context.strip()

            figures_saved.append({
                "index":        pic_idx,
                "page":         page_no,
                "filename":     filename,
                "status":       "saved",
                "size":         [w, h],
                "self_ref":     picture.self_ref,
                "docling_context": docling_context,
                "caption_text": None,
            })
            print(f"  Figure {pic_idx:03d} (p{page_no}) {w}x{h} → {filename}")

        except Exception as e:
            print(f"  WARNING: figure {pic_idx} skipped — {e}")

    # ── 4. Extraction log ─────────────────────────────────────────
    num_pages = doc.num_pages() if hasattr(doc, "num_pages") else None
    table_count = len(list(doc.tables))

    log = {
        "doc_id":        doc_id,
        "pdf":           str(pdf_path.resolve().relative_to(_ROOT.resolve())),   # relative path — portable
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
            all_logs.append({"doc_id": _get_doc_id(pdf_path),
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