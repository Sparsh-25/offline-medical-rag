"""
caption.py — Ollama/VLM figure captioning pipeline
============================================================================
Extracts a plain-language clinical summary (plus, where the figure is a
table/flowchart/pathway diagram, a markdown fragment) from each figure and
injects it into the document markdown so it becomes retrievable RAG content.

Output per figure → RAG-ready markdown:
  ![{clinical summary}](figures/filename.png)
  <!-- FIGURE_METADATA: {figure_id, figure_type, extraction_status} -->
  {enriched markdown for tables / flowcharts / pathway diagrams, if applicable}

Scope note: the extraction schema is intentionally minimal — only fields
that are actually consumed downstream (the embeddable summary text) are
requested from the VLM. Structured numeric extraction (hazard ratios,
p-values, cohort sizes, etc.) was removed because nothing in the pipeline
reads it back out yet (see decisions.md D10) — reintroduce it once a
consumer (e.g. metadata-filtered retrieval, numeric self-verification)
exists, ideally paired with a VLM capable of producing it reliably.
"""

import json
import re
import time
import yaml
import requests
import base64
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Optional, Any

_ROOT = Path(__file__).parent.parent
cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))

# ── VLM backend config (config.yaml → vlm section, sensible defaults) ────────
_VLM_CFG     = cfg.get("vlm", {})
VLM_MODEL    = _VLM_CFG.get("model",    "llava")
VLM_TIMEOUT  = int(_VLM_CFG.get("timeout",  300))
VLM_BASE_URL = _VLM_CFG.get("base_url", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VLM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

VLM_PROMPT = (
    "You are a Clinical Oncology Figure Extraction AI. Analyze this oncology figure "
    "and its surrounding text context.\n\n"

    "Figure ID: {figure_id}\n"
    "Context (caption / surrounding paragraph): \"{docling_extracted_context}\"\n\n"

    "Use BOTH the image and the text context. If they conflict, prefer the text. "
    "Do not invent numbers that are not visible in the image or context.\n\n"

    "Return ONLY the JSON object below. No markdown fences, no explanations.\n"
    "{\n"
    '  "figure_id": "string",\n'
    '  "figure_type": "one short label, e.g. Kaplan-Meier, Forest Plot, Bar Chart, '
    'Line Graph, Table, Flowchart, CONSORT Diagram, Biological Pathway, Medical Imaging, Other",\n'
    '  "summary": "3-4 sentences: (a) what the figure shows, (b) key numeric finding if visible '
    '(e.g. hazard ratio, p-value, median survival, cohort sizes), (c) clinical implication",\n'
    '  "clinical_relevance": "one concise sentence on why this finding matters, or null",\n'
    '  "enriched_text": "ONLY if this figure is a table, flowchart/CONSORT diagram, or biological '
    'pathway diagram: the extracted content as GitHub-flavored markdown (a table, a numbered flow '
    'list, or an entity/relation list). Otherwise null."\n'
    "}\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PYDANTIC SCHEMA
#
# Deliberately minimal — only fields the pipeline actually consumes today.
# See decisions.md D10 for why structured numeric fields (hazard_ratio,
# p_value, cohorts, data_points, etc.) were removed rather than kept unused.
# ─────────────────────────────────────────────────────────────────────────────

class VisualExtraction(BaseModel):
    figure_id:          str
    figure_type:        str            = "Other"
    summary:            str            = ""
    clinical_relevance: Optional[str]  = None
    enriched_text:      Optional[str]  = None   # markdown, only for tables/flowcharts/pathways


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METADATA BUILDER  — small tag dict embedded alongside each caption
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(extraction: VisualExtraction) -> dict:
    return {
        "figure_id":   extraction.figure_id,
        "figure_type": extraction.figure_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXTRACTION STATUS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def get_extraction_status(extraction: VisualExtraction) -> str:
    return "success" if extraction.summary.strip() else "failed"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  JSON HEALER  — strips LLM conversational noise, isolates JSON object
# ─────────────────────────────────────────────────────────────────────────────

def heal_json(raw: str) -> str:
    try:
        raw = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL)
        raw = raw.strip()
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1:
            return raw[start:end + 1]
        if raw:
            return "{" + raw.lstrip(",").rstrip(",") + "}"
    except Exception:
        pass
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 6.  INPUT SANITISER  — cleans VLM type-coercion problems pre-Pydantic
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_llava_output(data: dict, figure_id: str) -> dict:
    """Coerce raw VLM JSON into types Pydantic expects. Never raises."""

    data["figure_id"] = figure_id   # always override — never trust the model

    ft = data.get("figure_type")
    ft = ft.strip() if isinstance(ft, str) else None
    # Guard against the model echoing the prompt's own instruction text back
    # (observed live: "One short label, e.g. Kaplan-Meier, Forest Plot, ...")
    # instead of picking a value from it — a real label is short; the
    # instruction text is not.
    if not ft or len(ft) > 40 or "e.g." in ft.lower():
        ft = "Other"
    data["figure_type"] = ft

    summary = data.get("summary")
    if isinstance(summary, str):
        data["summary"] = summary
    else:
        data["summary"] = str(summary) if summary is not None else ""

    for sf in ("clinical_relevance", "enriched_text"):
        v = data.get(sf)
        if v is not None and not isinstance(v, str):
            v = str(v)
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            v = None
        data[sf] = v

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 7.  OLLAMA SERVER CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_ollama_server() -> bool:
    """
    Verify that Ollama daemon is running and VLM_MODEL is installed.
    Uses VLM_MODEL from config.yaml (vlm.model).
    """
    try:
        resp = requests.get(f"{VLM_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Match prefix so "llava:13b" and "llava-phi3" both satisfy "llava"
        model_prefix = VLM_MODEL.split(":")[0].lower()
        installed = any(m.lower().startswith(model_prefix) for m in models)
        if not installed:
            print(f"  [Ollama] WARNING: '{VLM_MODEL}' not found in registry.")
            print(f"  Run: ollama pull {VLM_MODEL}")
            return False
        print(f"  [Ollama] Server online — model '{VLM_MODEL}' verified ✓")
        return True
    except Exception as e:
        print(f"  [Ollama] Cannot connect to {VLM_BASE_URL}: {e}")
        print("  Ensure the Ollama application is running.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CORE VLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(
    server_online: bool,
    img_path: Path,
    figure_id: str,
    docling_context: str,
) -> Optional[VisualExtraction]:
    """
    Send one figure to the VLM via Ollama and return a validated VisualExtraction.
    Returns None on any failure (never raises).
    """
    try:
        if not server_online:
            return None

        with open(img_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")

        # Use .replace() — NOT .format() — because the prompt contains JSON schema
        # with curly braces that .format() mistakes for format variables and
        # raises KeyError.
        prompt_text = (
            VLM_PROMPT
            .replace("{figure_id}", str(figure_id))
            .replace("{docling_extracted_context}", str(docling_context or "No caption extracted."))
        )

        payload = {
            "model":  VLM_MODEL,
            "prompt": prompt_text,
            "images": [encoded],
            "stream": False,
            "format": "json",
        }

        response = requests.post(
            f"{VLM_BASE_URL}/api/generate",
            json=payload,
            timeout=VLM_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json().get("response", "").strip()
        if not raw:
            print(f"    [WARN] Empty response for {img_path.name}")
            return None

        fixed = heal_json(raw)

        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as je:
            print(f"    [DEBUG] JSONDecodeError: {je.msg!r} at pos {je.pos}")
            print(f"    [DEBUG] raw[:300]: {raw[:300]!r}")
            return None

        if not isinstance(data, dict):
            print(f"    [WARN] VLM returned non-dict JSON for {img_path.name}")
            return None

        data = _sanitise_llava_output(data, figure_id)

        try:
            extraction = VisualExtraction(**data)
        except ValidationError as ve:
            print(f"    [DEBUG] ValidationError ({ve.error_count()} errors) for {img_path.name}")
            for err in ve.errors()[:5]:
                print(f"      {err['loc']} → {err['msg']}")
            return None

        return extraction

    except Exception as e:
        print(f"    Extraction failed for {img_path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLACEHOLDER FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_placeholder_pattern(md_text: str) -> str:
    """
    Detect which figure placeholder format Docling used.

    Format inventory (in match-priority order):
      html_comment   — <!-- image ... -->          (Docling ≤ 1.x)
      empty_alt      — ![](path)                   (Docling image_mode="placeholder", some versions)
      image_alt      — ![image](path)              (Docling fallback, some versions)
      fallback_alt   — ![Figure from oncology document: …](path)  ← current Docling output
      any_alt_figure — ![anything](path/figure…)   (generic figure path)
    """
    if re.search(r"<!--\s*image", md_text, re.IGNORECASE):
        return "html_comment"
    if re.search(r"!\[\]\(", md_text):
        return "empty_alt"
    if re.search(r"!\[image\]", md_text, re.IGNORECASE):
        return "image_alt"
    # Current Docling output (image_mode="placeholder" in Docling ≥ 2.x)
    if re.search(r"!\[Figure from oncology document", md_text, re.IGNORECASE):
        return "fallback_alt"
    # Our own previously-injected summaries (re-run idempotency)
    if re.search(r"!\[[^\]]{10,}\]\(figures/[^)]+\.png\)", md_text):
        return "previous_output"
    if re.search(r"!\[[^\]]*\]\([^)]*figure[^)]*\)", md_text, re.IGNORECASE):
        return "any_alt_figure"
    # Unknown — print first image-like line to assist debugging
    for line in md_text.splitlines():
        if re.search(r"!\[|<img|figure", line, re.IGNORECASE):
            print(f"  [DEBUG] First unrecognised image line: {line!r}")
            break
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 9b. GROUNDING CHECK — flag drug-codename-shaped entities not in the source doc
#
# Narrow, explainable mitigation for the fabrication mode confirmed in
# decisions.md D14/D16: the VLM inventing a pharma codename (and an attached
# claim) that appears nowhere in the source document. Not a general fact-
# checker — deliberately scoped this narrow, see D16 for why.
# ─────────────────────────────────────────────────────────────────────────────

_DRUG_CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}-?\d{3,6}\b")

def _flag_ungrounded_entities(text: Optional[str], full_doc_text: str) -> list[str]:
    """Codename-shaped tokens in `text` that don't appear anywhere in `full_doc_text`."""
    if not text:
        return []
    candidates = set(_DRUG_CODE_PATTERN.findall(text))
    doc_lower  = full_doc_text.lower()
    return sorted(c for c in candidates if c.lower() not in doc_lower)


# ─────────────────────────────────────────────────────────────────────────────
# 10. REPLACEMENT MAP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_replacement_map(doc_dir: Path, server_online: bool) -> dict[str, Any]:
    """
    Caption every saved figure and return  self_ref → item  map.
    item = { summary, metadata, filename, extraction_data }
    """
    log_path = doc_dir / "extraction_log.json"
    if not log_path.exists():
        print(f"  No extraction_log.json in {doc_dir.name} — skipping")
        return {}

    log     = json.loads(log_path.read_text(encoding="utf-8"))
    figures = log.get("figures", [])
    if not figures:
        print("  No figures in extraction log")
        return {}

    fig_dir       = doc_dir / "figures"
    doc_md_path   = doc_dir / f"{doc_dir.name}.md"
    full_doc_text = doc_md_path.read_text(encoding="utf-8") if doc_md_path.exists() else ""
    replacement_map = {}

    print(f"  Captioning {len(figures)} figures...")

    for i, fig_info in enumerate(figures):

        # ── Skipped (too small / duplicate) → remove placeholder ─────────
        if fig_info.get("status") == "skipped":
            replacement_map[fig_info["self_ref"]] = {
                "action":          "remove",
                "summary":         None,
                "metadata":        None,
                "filename":        None,
                "extraction_data": None,
            }
            continue

        fig_path = fig_dir / fig_info["filename"]
        if not fig_path.exists():
            print(f"    Missing: {fig_info['filename']} — skipping")
            # Still need a map entry (even a no-op "remove" one) — its
            # placeholder still exists in the markdown, and without a
            # compensating entry every figure after it in this document
            # would shift by one position under the sequential-fallback
            # matcher in inject_captions_into_markdown() (see decisions.md D14).
            replacement_map[fig_info["self_ref"]] = {
                "action":          "remove",
                "summary":         None,
                "metadata":        None,
                "filename":        None,
                "extraction_data": None,
            }
            continue

        print(f"    [{i+1}/{len(figures)}] {fig_info['filename']} ", end="", flush=True)
        t0 = time.time()

        extraction = caption_image(
            server_online  = server_online,
            img_path       = fig_path,
            figure_id      = fig_info.get("self_ref", f"fig_{i}"),
            docling_context= fig_info.get("docling_context", ""),
        )
        elapsed = time.time() - t0

        if extraction:
            chunk_metadata            = build_metadata(extraction)
            status                    = get_extraction_status(extraction)
            chunk_metadata["extraction_status"] = status

            grounding_flags = _flag_ungrounded_entities(
                f"{extraction.summary} {extraction.clinical_relevance or ''}",
                full_doc_text,
            )
            if grounding_flags:
                chunk_metadata["grounding_flags"] = grounding_flags
                print(f"    [WARN] Possibly fabricated entit{'y' if len(grounding_flags)==1 else 'ies'} "
                      f"(not found in source document): {grounding_flags}")

            print(f"({elapsed:.1f}s) type={extraction.figure_type} status={status}")

            replacement_map[fig_info["self_ref"]] = {
                "summary":         extraction.summary,
                "metadata":        chunk_metadata,
                "filename":        fig_info["filename"],
                "extraction_data": extraction.model_dump(),
            }

            # Persist full extraction into the log
            fig_info["caption_text"]         = extraction.summary
            fig_info["clinical_extraction"]  = extraction.model_dump()
            fig_info["extraction_metadata"]  = chunk_metadata
            fig_info["extraction_status"]    = status

        else:
            # VLM failed — use filename-based fallback so placeholder is still replaced
            fallback_summary = f"Figure from oncology document: {fig_path.stem}"
            print(f"({elapsed:.1f}s) status=failed — using fallback text")
            replacement_map[fig_info["self_ref"]] = {
                "summary":         fallback_summary,
                "metadata":        {
                    "figure_id":         fig_info.get("self_ref", f"fig_{i}"),
                    "figure_type":       "Other",
                    "extraction_status": "failed",
                },
                "filename":        fig_info["filename"],
                "extraction_data": None,
            }
            fig_info["extraction_status"] = "failed"

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return replacement_map


# ─────────────────────────────────────────────────────────────────────────────
# 11. MARKDOWN INJECTOR
# ─────────────────────────────────────────────────────────────────────────────

def inject_captions_into_markdown(
    md_text:          str,
    replacement_map:  dict[str, Any],
    fig_dir_relative: str = "figures",
) -> tuple[str, int]:
    """
    Replace every figure placeholder in the markdown with:

      ![{VLM clinical summary}](figures/{filename}.png)
      <!-- FIGURE_METADATA: {figure_id, figure_type, extraction_status} -->
      {enriched table / flow / pathway markdown, if the VLM provided it}

    Handles all 5 placeholder formats and is idempotent on re-runs. First-time
    captioning matches bare <!-- image --> comments by explicit position;
    re-running on an already-captioned document matches by exact filename
    only (see decisions.md D20) — never by guessing.
    """
    replacements_made = 0

    # ── Detect first-time vs. re-run, BEFORE any mutation ────────────────────
    # A document that already has FIGURE_METADATA comments has been through
    # this function before — its original <!-- image --> placeholders are
    # already gone (consumed or deleted by that prior run), so there is
    # nothing left to positionally match against. Re-runs rely on exact
    # filename matching only (decisions.md D20).
    already_captioned = bool(re.search(r"<!--\s*FIGURE_METADATA:", md_text))

    # ── Pre-pass: normalise previously-injected output → empty-alt ───────────
    # On re-runs the markdown already contains our output; strip it so Format 2
    # can cleanly replace it again.
    known_filenames_pre = {
        item["filename"]
        for item in replacement_map.values()
        if item and item.get("filename")
    }

    def _strip_previous(m: re.Match) -> str:
        fname = Path(m.group(2)).name
        return f"![]({m.group(2)})" if fname in known_filenames_pre else m.group(0)

    md_text = re.sub(r"!\[([^\]]*)\]\((figures/[^)]+)\)", _strip_previous, md_text)
    md_text = re.sub(r"\n<!-- FIGURE_METADATA:[^\n]*-->", "", md_text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pop_item_by_filename(filename: str) -> Optional[dict]:
        """
        Consume and return a replacement_map entry matching filename — exact
        match only, no fallback. A figure that can't be exactly matched goes
        to orphan-recovery (still retrievable, just appended) rather than
        risk pairing the wrong caption with the wrong image via a guess.
        """
        for k in list(replacement_map.keys()):
            item = replacement_map[k]
            if item and item.get("filename") == filename:
                replacement_map[k] = None
                return item
        return None

    def _render(item: dict, default_filename: str = "figure.png") -> str:
        nonlocal replacements_made
        if item.get("action") == "remove":
            replacements_made += 1
            return ""

        summary    = item.get("summary") or "Figure from oncology document"
        fig_name   = item.get("filename") or default_filename
        meta_json  = json.dumps(item.get("metadata") or {}, ensure_ascii=False)

        output = (
            f"![{summary}]({fig_dir_relative}/{fig_name})\n"
            f"<!-- FIGURE_METADATA: {meta_json} -->"
        )

        # Append enriched markdown for tables, flowcharts, pathways — only
        # present when the VLM actually produced it for this figure type.
        ext_data = item.get("extraction_data")
        if ext_data and ext_data.get("enriched_text"):
            output += "\n" + ext_data["enriched_text"]

        replacements_made += 1
        return output

    # ── Format 1: bare <!-- image --> placeholders (first-time captioning only) ─
    # Docling never emits a usable ref attribute (confirmed directly against its
    # output — there is nothing to extract here), so there is no real ref to
    # match against. On a FRESH document, the Nth placeholder corresponds
    # exactly to the figure at index N in extraction_log.json — verified
    # directly that extract.py's doc.pictures enumeration (which builds that
    # log, skipped/filtered figures included) follows the same reading-order
    # traversal Docling's markdown export uses. That positional correspondence
    # only holds on a first-time run: once captioned, the placeholders are
    # gone (consumed or deleted), so a re-run has nothing left to count and
    # must not run this format at all (see already_captioned above).
    if not already_captioned:
        ordered_keys = list(replacement_map.keys())
        position = 0

        def _replace_html_comment(m: re.Match) -> str:
            nonlocal position
            if position >= len(ordered_keys):
                print(f"    [ERROR] More <!-- image --> placeholders in the markdown "
                      f"than figures in extraction_log.json ({len(ordered_keys)}) — "
                      f"document and log are out of sync; leaving the remainder "
                      f"unmatched rather than guessing.")
                return m.group(0)
            key = ordered_keys[position]
            position += 1
            item = replacement_map.get(key)
            if item is None:
                return m.group(0)
            replacement_map[key] = None
            return _render(item)

        md_text = re.sub(
            r"<!--\s*image[^>]*-->", _replace_html_comment, md_text, flags=re.IGNORECASE
        )

    # ── Format 2: ![]( path ) ── empty alt (and our pre-pass output) ─────────
    def _replace_empty_alt(m: re.Match) -> str:
        fname = Path(m.group(1)).name
        item  = _pop_item_by_filename(fname)
        return _render(item, default_filename=fname) if item else m.group(0)

    md_text = re.sub(r"!\[\]\(([^)]+)\)", _replace_empty_alt, md_text)

    # ── Format 3: ![image]( path ) ────────────────────────────────────────────
    def _replace_image_alt(m: re.Match) -> str:
        fname = Path(m.group(1)).name
        item  = _pop_item_by_filename(fname)
        return _render(item, default_filename=fname) if item else m.group(0)

    md_text = re.sub(
        r"!\[image\]\(([^)]+)\)", _replace_image_alt, md_text, flags=re.IGNORECASE
    )

    # ── Format 4: ![Figure from oncology document: …]( path ) ←── current Docling ─
    def _replace_fallback_2g(m: re.Match) -> str:
        fname = Path(m.group(2)).name
        item  = _pop_item_by_filename(fname)
        return _render(item, default_filename=fname) if item else m.group(0)

    md_text = re.sub(
        r"!\[(Figure from oncology document[^\]]*)\]\(([^)]+)\)",
        _replace_fallback_2g,
        md_text,
        flags=re.IGNORECASE,
    )

    # ── Format 5: catch-all — any ![alt](figures/known-file.png) ─────────────
    remaining_filenames = {
        item["filename"]
        for item in replacement_map.values()
        if item and item.get("filename")
    }

    if remaining_filenames:
        def _replace_any_guarded(m: re.Match) -> str:
            fname = Path(m.group(2)).name
            if fname not in remaining_filenames:
                return m.group(0)               # not one of ours — leave alone
            item = _pop_item_by_filename(fname)
            return _render(item, default_filename=fname) if item else m.group(0)

        md_text = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)", _replace_any_guarded, md_text
        )

    return md_text, replacements_made


# ─────────────────────────────────────────────────────────────────────────────
# 12. DOCUMENT-LEVEL ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def caption_document(doc_id: str, server_online: bool) -> dict:
    """Full caption pipeline for one document."""
    doc_dir  = Path(cfg["paths"]["extracted"]) / doc_id
    md_path  = doc_dir / f"{doc_id}.md"

    if not md_path.exists():
        print(f"  SKIP {doc_id} — no markdown found")
        return {"doc_id": doc_id, "status": "skipped"}

    print(f"\n{'─'*60}")
    print(f"Captioning: {doc_id}")

    replacement_map = build_replacement_map(doc_dir, server_online)
    if not replacement_map:
        print("  No figures to caption — markdown unchanged")
        return {"doc_id": doc_id, "status": "no_figures"}

    # Snapshot the "real" (non-removed) entry count BEFORE injection — the pop
    # helpers inside inject_captions_into_markdown() null out every matched
    # entry as a side effect, so counting after injection always yields 0
    # once matching succeeds (this masked failures/successes alike until now).
    real_entries = sum(
        1 for v in replacement_map.values()
        if v is not None and v.get("action") != "remove"
    )

    md_text = md_path.read_text(encoding="utf-8")
    fmt     = find_placeholder_pattern(md_text)
    print(f"  Placeholder format: {fmt}")

    updated_md, n_replaced = inject_captions_into_markdown(
        md_text, replacement_map, fig_dir_relative="figures"
    )
    md_path.write_text(updated_md, encoding="utf-8")

    print(f"  Replaced {n_replaced}/{real_entries} content placeholders ({len(replacement_map)} total map entries)")
    unmatched = real_entries - n_replaced
    if unmatched > 0:
        print(f"  WARNING: {unmatched} figure(s) had captions but no placeholder was matched")
        print(f"  Open {md_path} and search for remaining image syntax")

    # ── Recovery: figures whose placeholder was lost in a previous broken run ─
    # After injection, any real entry still non-None in the map means its
    # placeholder no longer exists in the markdown (consumed+lost by old buggy code).
    # Append them at the end of the document so they're not dropped from RAG.
    orphans = {
        ref: item for ref, item in replacement_map.items()
        if item is not None and item.get("action") != "remove"
    }
    if orphans:
        recovery = ["\n\n---\n\n## Extracted Figures\n"]
        for ref, item in orphans.items():
            summary   = item.get("summary") or "Figure from oncology document"
            fig_name  = item.get("filename") or "figure.png"
            meta_json = json.dumps(item.get("metadata") or {}, ensure_ascii=False)
            block = (
                f"\n![{summary}](figures/{fig_name})\n"
                f"<!-- FIGURE_METADATA: {meta_json} -->"
            )
            ext_data = item.get("extraction_data")
            if ext_data and ext_data.get("enriched_text"):
                block += "\n" + ext_data["enriched_text"]
            recovery.append(block)
            replacement_map[ref] = None
        updated_md += "".join(recovery)
        n_replaced += len(orphans)
        print(f"  Recovered {len(orphans)} figure(s) appended at end of document")
        md_path.write_text(updated_md, encoding="utf-8")

    return {
        "doc_id":               doc_id,
        "status":               "ok",
        "figures_captioned":    real_entries,
        "placeholders_replaced": n_replaced,
    }


def caption_all(doc_ids: Optional[list[str]] = None):
    """
    Caption every document in data/extracted/, or only the doc_ids given.
    Passing doc_ids lets you resume a partial/interrupted run (e.g. after
    the process was killed) without re-captioning documents already done —
    just pass the doc_ids that still need it.
    """
    ext_dir = Path(cfg["paths"]["extracted"])

    if doc_ids:
        doc_dirs = [ext_dir / d for d in doc_ids]
        missing  = [d.name for d in doc_dirs if not d.is_dir()]
        if missing:
            print(f"ERROR: unknown doc_id(s), no folder in {ext_dir}: {missing}")
            return
    else:
        doc_dirs = sorted(
            d for d in ext_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

    if not doc_dirs:
        print(f"No documents found in {ext_dir}")
        return

    server_online = verify_ollama_server()

    results = []
    for doc_dir in doc_dirs:
        result = caption_document(doc_dir.name, server_online)
        results.append(result)

    print(f"\n{'='*60}")
    total_figs     = sum(r.get("figures_captioned",    0) for r in results)
    total_replaced = sum(r.get("placeholders_replaced", 0) for r in results)
    print(f"Caption complete: {total_figs} figures captioned, {total_replaced} injected")
    for r in results:
        if r.get("status") == "ok":
            print(f"  ✓ {r['doc_id']}: {r['figures_captioned']} figs, "
                  f"{r['placeholders_replaced']} replaced")
        else:
            print(f"  ✗ {r['doc_id']}: {r['status']}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="caption.py — VLM figure captioning pipeline")
    p.add_argument(
        "--doc-id", action="append",
        help="Caption only this doc_id (repeatable: --doc-id a --doc-id b). "
             "Omit to caption every document in data/extracted/.",
    )
    args = p.parse_args()
    caption_all(doc_ids=args.doc_id)
