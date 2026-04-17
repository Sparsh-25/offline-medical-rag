import json
import re
import time
import yaml
import ssl
from pathlib import Path

# Fix for macOS Python SSL Certificate verification issues during moondream model download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

_ROOT = Path(__file__).parent.parent
cfg = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))

MEDICAL_PROMPT = (
    "You are an expert clinical oncologist analyzing a figure from a medical research paper. "
    "Provide a detailed, highly descriptive 3-4 sentence explanation of the visual findings. "
    "MANDATORY RESTRICTION: DO NOT start your response with conversational filler like 'This is an image of', 'The image shows', 'The figure displays', 'This picture appears to be'. Start immediately with the clinical facts! "
    "CRITICAL INSTRUCTIONS based on figure type: "
    "1. If a GRAPH (Kaplan-Meier, forest plot, bar chart): DO NOT just read axes. Describe the GEOMETRIC TREND. "
    "Are curves separating? Does one treatment arm stay above another over time? Which cohort visually performs best? "
    "2. If a FLOWCHART or CLINICAL PATHWAY (CONSORT diagram, decision tree): Trace the logical flow. "
    "Where is the largest patient drop-off? What are the key decision nodes or final cohort sizes? "
    "3. If a MECHANISTIC DIAGRAM or MEDICAL SCANS (cellular pathway, CT/MRI): Describe the structural, anatomical, or biological interactions explicitly shown. "
    "MANDATORY: Always explicitly state the specific figure type, the primary clinical endpoint or biological target, and any key numerical values (p-values, Hazard Ratios, n=) visible."
)


import requests
import base64

def verify_ollama_server():
    """
    Verify that the Ollama local daemon is running natively in the background 
    and the 'llava' model is available in the local registry.
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        llava_installed = any(m.startswith("llava") for m in models)
        
        if not llava_installed:
            print("  [Ollama] WARNING: 'llava' model not found in registry. Please run `ollama run llava` in terminal.")
            return False
            
        print("  [Ollama] Server online and 'llava' model verified successfully!")
        return True
    except Exception as e:
        print(f"  [Ollama] ERROR connecting to localhost:11434")
        print("  Please ensure you have installed the Ollama application and it is actively running!")
        return False


def caption_image(server_online: bool, img_path: Path) -> str:
    """
    Generate a medical caption for one figure by streaming a base64 encoded image
    natively to the local Ollama backend via HTTP POST.
    """
    try:
        if not server_online:
            return f"Figure from oncology document: {img_path.stem}"
            
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        payload = {
            "model": "llava",
            "prompt": MEDICAL_PROMPT,
            "images": [encoded_string],
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        
        caption = response.json().get("response", "").strip()
            
        if not caption:
            return f"Figure from oncology document: {img_path.stem}"
        return caption
    except Exception as e:
        print(f"    Caption failed for {img_path.name}: {e}")
        return f"Figure from oncology document: {img_path.stem}"


def find_placeholder_pattern(md_text: str) -> str:
    """
    Detect which placeholder format Docling used in this markdown.
    Returns the format name so the replacer knows what regex to use.
    """
    if re.search(r'<!--\s*image', md_text, re.IGNORECASE):
        return "html_comment"
    if re.search(r'!\[\]\(', md_text):
        return "empty_alt"
    if re.search(r'!\[image\]', md_text, re.IGNORECASE):
        return "image_alt"
    return "unknown"


def build_replacement_map(doc_dir: Path, server_online: bool) -> dict[str, str]:
    """
    For each saved figure PNG in extraction_log.json,
    generate a caption and return a map of self_ref → caption.
    
    This is the source of truth — we use self_ref (Docling's internal ID)
    to match figures to their placeholders in the markdown.
    """
    log_path = doc_dir / "extraction_log.json"
    if not log_path.exists():
        print(f"  No extraction_log.json in {doc_dir.name} — skipping captions")
        return {}

    log = json.loads(log_path.read_text(encoding="utf-8"))
    figures = log.get("figures", [])

    if not figures:
        print(f"  No figures recorded in extraction log")
        return {}

    fig_dir = doc_dir / "figures"
    replacement_map = {}   # self_ref → caption

    print(f"  Captioning {len(figures)} figures...")
    for i, fig_info in enumerate(figures):
        if fig_info.get("status") == "skipped":
            replacement_map[fig_info["self_ref"]] = {
                "action": "remove",
                "caption": None,
                "filename": None
            }
            continue

        fig_path = fig_dir / fig_info["filename"]

        if not fig_path.exists():
            print(f"    Missing: {fig_info['filename']} — skipping")
            continue

        print(f"    [{i+1}/{len(figures)}] {fig_info['filename']} ", end="", flush=True)
        t0 = time.time()

        caption = caption_image(server_online, fig_path)
        elapsed = time.time() - t0

        print(f"({elapsed:.1f}s)")
        print(f"    Caption: {caption[:120]}...")

        # Update the log with the caption
        fig_info["caption_text"] = caption
        replacement_map[fig_info["self_ref"]] = {
            "caption": caption,
            "filename": fig_info["filename"]
        }

    # Write updated log with captions filled in
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return replacement_map


def inject_captions_into_markdown(
    md_text: str,
    replacement_map: dict[str, str],
    fig_dir_relative: str = "figures"
) -> tuple[str, int]:
    """
    Replace Docling's figure placeholders with proper markdown image syntax
    containing the generated caption as alt-text.

    Returns (updated_markdown, number_of_replacements_made).

    Handles multiple placeholder formats Docling may produce.
    """
    replacements_made = 0

    # ── Format 1: HTML comment style ─────────────────────────────
    # <!-- image, filename: pictures/picture-3.png -->
    def replace_html_comment(match):
        nonlocal replacements_made
        raw = match.group(0)

        # Try to extract the self_ref / filename hint from the comment
        filename_match = re.search(r'filename:\s*([^\s,>]+)', raw)
        ref_match = re.search(r'ref:\s*([^\s,>]+)', raw)

        caption = None
        fig_name = None
        matched_ref = None

        # Try matching by self_ref first (most reliable)
        if ref_match:
            matched_ref = ref_match.group(1)
            item = replacement_map.get(matched_ref)
            if item:
                caption = item["caption"]
                fig_name = item["filename"]

        # Fall back: try matching by position/index in replacement_map
        if caption is None and replacement_map:
            # Take the next unused item in order
            for ref, item in replacement_map.items():
                if item is not None:
                    matched_ref = ref
                    replacement_map[ref] = None  # mark as used
                    
                    if item.get("action") == "remove":
                        replacements_made += 1
                        return ""  # Remove the placeholder!

                    caption = item.get("caption")
                    fig_name = item.get("filename")
                    break

        if caption is None:
            return raw  # leave placeholder if no caption

        if fig_name is None:
            fig_name = filename_match.group(1).split("/")[-1] if filename_match else "figure.png"

        replacements_made += 1
        return f"![{caption}]({fig_dir_relative}/{fig_name})"

    md_text = re.sub(
        r'<!--\s*image[^>]*-->',
        replace_html_comment,
        md_text,
        flags=re.IGNORECASE
    )

    # ── Format 2: Empty alt-text  ![](path/to/figure.png) ────────
    def replace_empty_alt(match):
        nonlocal replacements_made
        img_path_in_md = match.group(1)

        # Extract just the filename to look up in our figures
        fig_filename = Path(img_path_in_md).name

        # Find matching item from replacement_map by filename
        caption = None
        for ref, item in replacement_map.items():
            if item and item["filename"] == fig_filename:
                caption = item["caption"]
                replacement_map[ref] = None
                break

        # Sequential fallback
        if caption is None:
            for ref, item in replacement_map.items():
                if item is not None:
                    replacement_map[ref] = None
                    if item.get("action") == "remove":
                        replacements_made += 1
                        return ""
                    caption = item.get("caption")
                    break

        if caption is None:
            return match.group(0)

        replacements_made += 1
        return f"![{caption}]({fig_dir_relative}/{fig_filename})"

    md_text = re.sub(
        r'!\[\]\(([^)]+)\)',
        replace_empty_alt,
        md_text
    )

    # ── Format 3: ![image](path) ──────────────────────────────────
    def replace_image_alt(match):
        nonlocal replacements_made
        img_path_in_md = match.group(1)
        fig_filename = Path(img_path_in_md).name

        caption = None
        for ref, item in replacement_map.items():
            if item and item["filename"] == fig_filename:
                caption = item["caption"]
                replacement_map[ref] = None
                break

        if caption is None:
            for ref, item in replacement_map.items():
                if item is not None:
                    replacement_map[ref] = None
                    if item.get("action") == "remove":
                        replacements_made += 1
                        return ""
                    caption = item.get("caption")
                    break

        if caption is None:
            return match.group(0)

        replacements_made += 1
        return f"![{caption}]({fig_dir_relative}/{fig_filename})"

    md_text = re.sub(
        r'!\[image\]\(([^)]+)\)',
        replace_image_alt,
        md_text,
        flags=re.IGNORECASE
    )

    return md_text, replacements_made


def caption_document(doc_id: str, server_online: bool) -> dict:
    """
    Full caption pipeline for one document.
    1. Generate captions for all saved figures
    2. Inject captions into markdown as alt-text
    3. Save updated markdown
    """
    doc_dir = Path(cfg["paths"]["extracted"]) / doc_id
    md_path = doc_dir / f"{doc_id}.md"

    if not md_path.exists():
        print(f"  SKIP {doc_id} — no markdown found")
        return {"doc_id": doc_id, "status": "skipped"}

    print(f"\nCaptioning: {doc_id}")

    # Step 1: generate captions, get self_ref → caption map
    replacement_map = build_replacement_map(doc_dir, server_online)

    if not replacement_map:
        print(f"  No figures to caption — markdown unchanged")
        return {"doc_id": doc_id, "status": "no_figures"}

    # Step 2: inject into markdown
    md_text = md_path.read_text(encoding="utf-8")
    fmt = find_placeholder_pattern(md_text)
    print(f"  Placeholder format detected: {fmt}")

    updated_md, n_replaced = inject_captions_into_markdown(
        md_text,
        replacement_map,
        fig_dir_relative="figures"
    )

    # Step 3: save
    md_path.write_text(updated_md, encoding="utf-8")

    print(f"  Replaced {n_replaced}/{len(replacement_map)} placeholders")

    # Warn if some placeholders were not matched
    unmatched = len(replacement_map) - n_replaced
    if unmatched > 0:
        print(f"  WARNING: {unmatched} figures have captions but no placeholder was found")
        print(f"  This means the markdown uses a format the replacer did not catch")
        print(f"  Open {md_path} and search for remaining image syntax manually")

    return {
        "doc_id": doc_id,
        "status": "ok",
        "figures_captioned": len(replacement_map),
        "placeholders_replaced": n_replaced,
    }


def caption_all():
    ext_dir = Path(cfg["paths"]["extracted"])
    doc_dirs = sorted(
        d for d in ext_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    if not doc_dirs:
        print(f"No documents found in {ext_dir}")
        return

    # Verify that the Ollama backend is running natively
    server_online = verify_ollama_server()

    results = []
    for doc_dir in doc_dirs:
        result = caption_document(doc_dir.name, server_online)
        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    total_figs = sum(r.get("figures_captioned", 0) for r in results)
    total_replaced = sum(r.get("placeholders_replaced", 0) for r in results)
    print(f"Caption complete: {total_figs} figures captioned, {total_replaced} injected into markdown")

    for r in results:
        status = r.get("status")
        if status == "ok":
            print(f"  {r['doc_id']}: {r['figures_captioned']} figures, {r['placeholders_replaced']} replaced")
        else:
            print(f"  {r['doc_id']}: {status}")


if __name__ == "__main__":
    caption_all()