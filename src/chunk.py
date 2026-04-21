import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

cfg = yaml.safe_load(open("config.yaml"))

# ── Recency weight computation ────────────────────────────────────

def compute_recency_weight(year: Optional[int]) -> float:
    if year is None:
        return 0.7
    base = cfg["recency"]["baseline_year"]   # 2015
    peak = cfg["recency"]["peak_year"]       # 2024
    if year <= base: return 0.5
    if year >= peak: return 1.0
    return round(0.5 + 0.5 * (year - base) / (peak - base), 3)

# ── Chunk dataclass — every field the LLM and retriever will ever see ──

@dataclass
class Chunk:
    # identity
    chunk_id: str
    doc_id: str

    # ── FROM meta.json ──────────────────────────────────────────
    title: Optional[str]
    pub_year: Optional[int]
    pub_month: Optional[int]
    recency_weight: float          # computed from pub_year
    source_type: Optional[str]     # rct | guideline | meta_analysis | review | mechanism | staging
    journal: Optional[str]
    doi: Optional[str]
    guideline_version: Optional[str]
    cancer_type: Optional[str]
    cancer_subtype: Optional[list]
    population: Optional[str]
    line_of_therapy: Optional[str]
    drug_focus: Optional[list]
    drug_class: Optional[str]

    # ── FROM docling markdown (structural) ─────────────────────
    section_h1: Optional[str]
    section_h2: Optional[str]
    section_h3: Optional[str]
    content_type: str              # prose | table | figure_caption
    page_hint: Optional[int]       # approximate, from extraction_log

    # ── THE ACTUAL TEXT ─────────────────────────────────────────
    chunk_text: str
    token_count: int


# ── Load meta.json for a document ────────────────────────────────

def load_meta(doc_id: str) -> dict:
    meta_path = (
        Path(cfg["paths"]["extracted"]) / doc_id / "meta.json"
    )
    if not meta_path.exists():
        print(f"  WARNING: no meta.json for {doc_id} — metadata will be null")
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ── Detect content type from chunk text ──────────────────────────

def detect_content_type(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("|") and "|" in stripped[1:]:
        return "table"
    if stripped.startswith("!["):
        return "figure_caption"
    return "prose"


def rough_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ── Header-aware splitter ─────────────────────────────────────────

def split_markdown_by_headers(md_text: str) -> list[dict]:
    """
    Splits markdown into sections preserving header hierarchy.
    Returns list of dicts: {level, heading, body, approx_position}
    """
    header_re = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(header_re.finditer(md_text))

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        sections.append({
            "level": len(m.group(1)),
            "heading": m.group(2).strip(),
            "body": body,
            "char_position": m.start(),   # rough position for page hint
        })

    return sections


# ── Sub-split oversized sections on paragraph boundaries ──────────

def subsplit(body: str, max_tokens: int) -> list[str]:
    if rough_tokens(body) <= max_tokens:
        return [body]

    paragraphs = re.split(r'\n\n+', body)
    chunks = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if rough_tokens(candidate) > max_tokens and current:
            chunks.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── Main chunker ──────────────────────────────────────────────────

def chunk_document(doc_id: str) -> list[Chunk]:
    doc_dir = Path(cfg["paths"]["extracted"]) / doc_id
    md_path = doc_dir / f"{doc_id}.md"

    if not md_path.exists():
        print(f"  SKIP {doc_id} — markdown not found")
        return []

    # ── Load both sources ──────────────────────────────────────
    md_text = md_path.read_text(encoding="utf-8")
    meta = load_meta(doc_id)          # may be {} if missing

    # ── Compute derived fields from meta ───────────────────────
    pub_year = meta.get("pub_year")
    recency_weight = compute_recency_weight(pub_year)

    # ── Walk sections ──────────────────────────────────────────
    sections = split_markdown_by_headers(md_text)

    chunks = []
    chunk_index = 0
    h1 = h2 = h3 = None

    # Approximate chars-per-page for page hint (rough)
    doc_char_len = len(md_text)

    for sec in sections:
        level, heading, body = sec["level"], sec["heading"], sec["body"]

        # Update header breadcrumb
        if level == 1:
            h1, h2, h3 = heading, None, None
        elif level == 2:
            h2, h3 = heading, None
        elif level == 3:
            h3 = heading

        if not body:
            continue

        # Sub-split if needed
        sub_bodies = subsplit(body, cfg["chunking"]["max_chunk_tokens"])

        for sub_body in sub_bodies:
            tokens = rough_tokens(sub_body)
            if tokens < cfg["chunking"]["min_chunk_tokens"]:
                continue

            chunk = Chunk(
                # identity
                chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
                doc_id=doc_id,

                # ── from meta.json — baked in at build time ──
                title=meta.get("title"),
                pub_year=pub_year,
                pub_month=meta.get("pub_month"),
                recency_weight=recency_weight,
                source_type=meta.get("source_type"),
                journal=meta.get("journal"),
                doi=meta.get("doi"),
                guideline_version=meta.get("guideline_version"),
                cancer_type=meta.get("cancer_type"),
                cancer_subtype=meta.get("cancer_subtype"),
                population=meta.get("population"),
                line_of_therapy=meta.get("line_of_therapy"),
                drug_focus=meta.get("drug_focus"),
                drug_class=meta.get("drug_class"),

                # ── from markdown structure ──────────────────
                section_h1=h1,
                section_h2=h2,
                section_h3=h3,
                content_type=detect_content_type(sub_body),
                page_hint=None,   # filled below if extraction_log exists

                # ── content ──────────────────────────────────
                chunk_text=sub_body,
                token_count=tokens,
            )
            chunks.append(chunk)
            chunk_index += 1

    # ── Best-effort page hints from extraction_log ─────────────
    # The log has figure positions; for prose we interpolate
    log_path = doc_dir / "extraction_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        total_pages = log.get("pages") or 1
        for chunk in chunks:
            # rough: char position fraction × total pages
            # (not exact but useful for citation display)
            pass   # leave as None for now — implement in Phase 1 if needed

    return chunks


# ── Process all documents ─────────────────────────────────────────

def chunk_all():
    ext_dir = Path(cfg["paths"]["extracted"])
    out_path = Path(cfg["paths"]["chunks"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for doc_dir in sorted(ext_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if doc_id.startswith("_"):      # skip _extraction_summary etc
            continue

        print(f"\nChunking: {doc_id}")
        chunks = chunk_document(doc_id)
        all_chunks.extend(chunks)

        # Per-doc stats
        types = {}
        for c in chunks:
            types[c.content_type] = types.get(c.content_type, 0) + 1
        print(f"  {len(chunks)} chunks — {types}")

    # Write jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    # Write chunk_map — used at query time for instant lookup
    chunk_map = {c.chunk_id: asdict(c) for c in all_chunks}
    map_path = Path("index/chunk_map.json")
    map_path.parent.mkdir(exist_ok=True)
    map_path.write_text(
        json.dumps(chunk_map, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nTotal: {len(all_chunks)} chunks → {out_path}")
    print(f"Chunk map → index/chunk_map.json")

    # Quick validation — flag any chunk missing critical metadata
    missing_meta = [
        c for c in all_chunks
        if c.source_type is None or c.pub_year is None
    ]
    if missing_meta:
        print(f"\nWARNING: {len(missing_meta)} chunks missing source_type or pub_year")
        print("  Docs affected:", set(c.doc_id for c in missing_meta))
        print("  Fill in meta.json and re-run chunk_all()")

    return all_chunks


if __name__ == "__main__":
    chunk_all()