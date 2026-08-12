"""
meta_builder.py v2.0 — Defensive clinical metadata extraction pipeline
=======================================================================
Phase 1 (Offline/Deterministic):
  a) OFFLINE_GROUND_TRUTH  — hardcoded, highest authority
  b) CrossRef API          — title, journal, pub_year, pub_month
  c) PubMed/Entrez API     — abstract, MeSH, pub_types

Phase 2 (LLM Semantic):
  Groq (llama-3.1-8b-instant, 14,400 req/day free) or Ollama (local, unlimited)

Phase 3 (Defensive Merge):
  Rule 1: Offline values NEVER overwritten by LLM.
  Rule 2: LLM may only FILL null / [] offline fields.
  Rule 3: Any offline↔LLM conflict is logged in `_conflicts` + sets `_needs_review`.
  Rule 4: `_needs_review` = True if any conflict OR confidence < 0.85.
Schema v2 changes vs v1:
  - drug_focus    : list[str]   (was drug: str)
  - cancer_subtype: list[str]   (was str | null)
  - cancer_type   : "Breast"    (fixed, not snake_case)
  - _conflicts    : list[dict]  (logged field-level conflicts)
"""

import json
import os
import re
import ssl
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import certifi
import pypdfium2 as pdfium
import yaml
from Bio import Entrez


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & SECRETS
# Secrets are loaded from .secrets/secrets.yaml (gitignored).
# Fallback: GROQ_API_KEY / ENTREZ_EMAIL environment variables.
# ═══════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        result[k] = _deep_merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


def _load_cfg() -> dict:
    root = Path(__file__).parent.parent
    cfg  = yaml.safe_load((root / "config.yaml").read_text())

    # Layer 1 — .secrets/secrets.yaml (recommended, gitignored)
    secrets_path = root / ".secrets" / "secrets.yaml"
    if secrets_path.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(secrets_path.read_text()))

    # Layer 2 — environment variables (CI/CD / Docker)
    if os.environ.get("GROQ_API_KEY"):
        cfg.setdefault("api", {}).setdefault("groq", {})["api_key"] = os.environ["GROQ_API_KEY"]
    if os.environ.get("ENTREZ_EMAIL"):
        cfg.setdefault("api", {})["entrez_email"] = os.environ["ENTREZ_EMAIL"]

    return cfg


CFG           = _load_cfg()
API_CFG       = CFG.get("api", {})
EXTRACTED_DIR = Path(CFG["paths"]["extracted"])
RAW_DIR       = Path(CFG["paths"]["raw"])
SSL_CTX       = ssl.create_default_context(cafile=certifi.where())
Entrez.email  = API_CFG.get("entrez_email", "your@email.com")


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA v2.0
# ═══════════════════════════════════════════════════════════════════════

VALID_SOURCE_TYPES = {"guideline", "meta_analysis", "rct", "retrospective", "mechanism", "review"}
LIST_FIELDS        = {"cancer_subtype", "drug_focus"}  # always list[str]


def _enforce_type(field: str, value: Any) -> Any:
    """Coerce a value to the correct schema type for the given field."""
    if field in LIST_FIELDS:
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            if s.lower() in ("null", "none", ""):
                return []
            return [v.strip() for v in s.split(",") if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value
                    if v and str(v).lower() not in ("null", "none", "")]
        return []
    if field == "cancer_type":
        return "Breast"

    return value


def _blank_meta(doc_id: str) -> dict:
    return {
        "doc_id":            doc_id,
        "title":             None,
        "pub_year":          None,
        "pub_month":         None,
        "source_type":       None,
        "journal":           None,
        "doi":               None,
        "guideline_version": None,
        "cancer_type":       "Breast",
        "cancer_subtype":    [],        # list[str]
        "population":        None,
        "line_of_therapy":   None,
        "drug_focus":        [],        # list[str]
        "drug_class":        None,
        "notes":             None,
        "_needs_review":     True,
        "_llm_confidence":   None,
        "_conflicts":        [],
    }


# ═══════════════════════════════════════════════════════════════════════
# OFFLINE GROUND TRUTH  (hardcoded — highest authority for known documents)
# ═══════════════════════════════════════════════════════════════════════

OFFLINE_GROUND_TRUTH: dict[str, dict] = {
    # "nccn_breast_v2_2026": {
    #     "source_type":    "guideline",
    #     "line_of_therapy": "all",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HR+", "HER2+", "TNBC", "HER2-low"],
    # },
    # "asco_her2_testing_guideline_2023": {
    #     "source_type":    "guideline",
    #     "line_of_therapy": "all",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HER2+", "HER2-low"],
    # },
    # "monaleesa2_subanalysis_2018": {
    #     "source_type":    "rct",
    #     "line_of_therapy": "first_line",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HR+", "HER2-"],
    #     "drug_focus":     ["ribociclib", "letrozole"],
    #     "drug_class":     "CDK4/6 inhibitor",
    # },
    # "sci_reports_nact_hr_pos_2025": {
    #     "source_type":    "retrospective",
    #     "line_of_therapy": "neoadjuvant",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HR+", "HER2-"],
    # },
    # "frontiers_molbiosci_multiomics_2022": {
    #     "source_type":    "review",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HR+", "HER2+", "TNBC"],
    # },
    # "frontiers_pubhealth_tdxd_cea_2023": {
    #     "source_type":    "review",
    #     "line_of_therapy": "second_line",
    #     "cancer_type":    "Breast",
    #     "cancer_subtype": ["HER2-low"],
    #     "drug_focus":     ["trastuzumab deruxtecan"],
    #     "drug_class":     "antibody-drug conjugate",
    # },
}

# PDF filename stem → clean doc_id
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


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1a — PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

_DOI_RE   = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_NCCN_VER = re.compile(r"version\s+(\d+\.\d{4})", re.IGNORECASE)


def extract_pdf_text(pdf_path: Path, max_pages: int = 4) -> str:
    doc   = pdfium.PdfDocument(str(pdf_path))
    pages = [doc[i].get_textpage().get_text_range() for i in range(min(max_pages, len(doc)))]
    doc.close()
    return unicodedata.normalize("NFKC", "\n".join(pages))


def _find_doi(text: str) -> Optional[str]:
    hits = _DOI_RE.findall(text)
    return hits[0].rstrip(".,;)") if hits else None


def _find_nccn_version(text: str) -> Optional[str]:
    m = _NCCN_VER.search(text[:3000])
    return m.group(0) if m else None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1b — CROSSREF API
# ═══════════════════════════════════════════════════════════════════════

def fetch_crossref_meta(doi: str) -> dict:
    if not doi:
        return {}
    try:
        req = urllib.request.Request(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "oncology-rag/2.0"},
        )
        with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as r:
            msg = json.loads(r.read()).get("message", {})
    except Exception as e:
        print(f"    CrossRef error: {e}")
        return {}

    result: dict = {}
    if msg.get("title"):          result["title"]   = msg["title"][0]
    if msg.get("container-title"): result["journal"] = msg["container-title"][0]
    for key in ("published-print", "published-online", "created"):
        parts = msg.get(key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            result["pub_year"] = int(parts[0][0])
            if len(parts[0]) >= 2:
                result["pub_month"] = int(parts[0][1])
            break
    if "DOI" in msg:
        result["doi"] = msg["DOI"]
    return result


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1c — PUBMED / ENTREZ API
# ═══════════════════════════════════════════════════════════════════════

def _http_xml(url: str) -> Optional[ET.Element]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "oncology-rag/2.0"})
        with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as r:
            return ET.fromstring(r.read())
    except Exception as e:
        print(f"    HTTP error: {e}")
        return None


def search_pubmed_by_doi(doi: str) -> Optional[str]:
    if not doi:
        return None
    root = _http_xml(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={urllib.parse.quote(doi)}[doi]"
        f"&retmax=1&retmode=xml&email={urllib.parse.quote(Entrez.email)}"
    )
    ids = root.findall(".//Id") if root is not None else []
    return ids[0].text.strip() if ids else None


def fetch_pubmed_meta(pmid: str) -> dict:
    root = _http_xml(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&rettype=xml&retmode=xml"
        f"&email={urllib.parse.quote(Entrez.email)}"
    )
    if root is None:
        return {}

    title_el = root.find(".//ArticleTitle")
    title    = "".join(title_el.itertext()).strip() if title_el is not None else None

    def txt(path: str) -> Optional[str]:
        el = root.find(path)
        return el.text.strip() if el is not None and el.text else None

    doi = None
    for aid in root.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text.strip()

    pub_year = pub_month = None
    for dp in (".//PubMedPubDate[@PubStatus='pubmed']",
               ".//PubMedPubDate[@PubStatus='entrez']",
               ".//ArticleDate"):
        d = root.find(dp)
        if d is not None:
            y, m = d.find("Year"), d.find("Month")
            if y is not None: pub_year = int(y.text)
            if m is not None:
                try: pub_month = int(m.text)
                except ValueError: pass
            break

    return {
        "title":      title,
        "journal":    txt(".//Journal/Title") or txt(".//ISOAbbreviation"),
        "doi":        doi,
        "pub_year":   pub_year,
        "pub_month":  pub_month,
        "abstract":   " ".join(el.text or "" for el in root.findall(".//AbstractText")).strip() or None,
        "mesh_terms": [el.text for el in root.findall(".//MeshHeading/DescriptorName") if el.text],
        "pub_types":  [el.text for el in root.findall(".//PublicationType") if el.text],
    }


def _infer_cancer_type(mesh: list) -> Optional[str]:
    j = " ".join(mesh).lower()
    if "breast neoplasm" in j:   return "Breast"
    if "lung neoplasm"   in j:   return "Lung"
    if "colorectal"      in j:   return "Colorectal"
    if "prostate"        in j:   return "Prostate"
    if "ovarian"         in j:   return "Ovarian"
    return None


_PUB_TYPE_MAP = [
    ("randomized controlled trial",  "rct"),
    ("clinical trial, phase iii",    "rct"),
    ("clinical trial, phase ii",     "rct"),
    ("meta-analysis",                "meta_analysis"),
    ("systematic review",            "review"),
    ("practice guideline",           "guideline"),
]


def _infer_source_type(pub_types: list) -> Optional[str]:
    j = " ".join(pub_types).lower()
    for pattern, st in _PUB_TYPE_MAP:
        if pattern in j:
            return st
    return "review" if "review" in j else None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — LLM SEMANTIC EXTRACTION
# Prompt v2.0 per spec — fixes: lazy nulls, monotherapy bias,
# study design hallucination, ontology drift.
# ═══════════════════════════════════════════════════════════════════════

PROMPT_V2 = """\
You are a Lead Clinical Oncology Engineer. Extract and format clinical metadata into a strict JSON schema.

CRITICAL DIRECTIVES:
1. Biomarkers (`cancer_subtype`): NEVER leave empty if deducible. Translate "hormone receptor positive" to "HR+", "HER2-negative" to "HER2-". \
For broad mechanism reviews covering the whole disease, output ["HR+", "HER2+", "TNBC", "HER2-low"]. ALWAYS an array of strings.
2. Combination Therapy (`drug_focus`): Look for "plus", "and", "+". Capture ALL drugs. ALWAYS an array. If no specific drug studied, output [].
3. Accurate Study Design (`source_type`): Do NOT hallucinate RCTs. Retrospective database analyses (SEER, registry, claims) are "retrospective". \
Cost-effectiveness/economic models are "review". Valid values ONLY: ["guideline", "meta_analysis", "rct", "retrospective", "mechanism", "review"].
4. Population (`population`): Only specific trial inclusion criteria. No epidemiology. null if none.
5. Ontology: `cancer_type` MUST always be exactly "Breast".

OUTPUT — respond with ONLY this JSON object, no markdown fences, no commentary:
{{
  "title": "string or null",
  "source_type": "one of the six valid values",
  "cancer_type": "Breast",
  "cancer_subtype": ["array of strings"],
  "population": "string or null",
  "line_of_therapy": "first_line|second_line|third_line_plus|adjuvant|neoadjuvant|all|null",
  "drug_focus": ["array of drug names"],
  "drug_class": "string or null",
  "notes": "one sentence on clinical significance",
  "_llm_confidence": float
}}

TEXT:
{context}
"""


def _build_context(pdf_text: str, abstract: Optional[str]) -> str:
    parts = []
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}")
    parts.append(f"FIRST-PAGE TEXT:\n{pdf_text[:1500]}")
    return "\n\n".join(parts)


def _extract_via_groq(context: str) -> dict:
    from groq import Groq
    cfg    = API_CFG.get("groq", {})
    client = Groq(api_key=cfg["api_key"])
    resp   = client.chat.completions.create(
        model=cfg.get("model", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": PROMPT_V2.format(context=context)}],
        temperature=0.0,
        max_tokens=700,
        response_format={"type": "json_object"},   # enforces valid JSON
    )
    return json.loads(resp.choices[0].message.content)


def _extract_via_ollama(context: str) -> dict:
    cfg     = API_CFG.get("ollama", {})
    payload = json.dumps({
        "model":   cfg.get("model", "llama3.1"),
        "prompt":  PROMPT_V2.format(context=context),
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{cfg.get('base_url', 'http://localhost:11434')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.loads(r.read())["response"])


def run_llm_extraction(pdf_text: str, abstract: Optional[str]) -> dict:
    """Route to Groq or Ollama. Returns raw dict; sets _error key on failure."""
    backend = API_CFG.get("llm_backend", "groq").lower()
    context = _build_context(pdf_text, abstract)
    try:
        if backend == "groq":   return _extract_via_groq(context)
        if backend == "ollama": return _extract_via_ollama(context)
        return {"_llm_confidence": 0.0, "_error": f"Unknown backend: {backend}"}
    except Exception as e:
        return {"_llm_confidence": 0.0, "_error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — DEFENSIVE MERGE
# ═══════════════════════════════════════════════════════════════════════

# Fields handled separately — never touched by the merge loop
_SKIP_MERGE = {"doc_id", "_needs_review", "_llm_confidence", "_conflicts"}


def _is_empty(val: Any) -> bool:
    if val is None or val == "" or val == []:
        return True
    return isinstance(val, list) and all(v is None for v in val)


def _normalize(val: Any):
    if isinstance(val, list):
        return sorted(str(v).lower().strip() for v in val if v)
    if isinstance(val, str):
        return val.lower().strip()
    return val


def defensive_merge(offline: dict, llm: dict) -> dict:
    """
    Merge llm into offline with strict supremacy rules.
    Returns the merged dict (offline is mutated in-place and returned).
    """
    conflicts: list[dict] = []

    for field, llm_val in llm.items():
        if field in _SKIP_MERGE or field.startswith("_"):
            continue

        llm_val = _enforce_type(field, llm_val)   # enforce schema types first
        off_val = offline.get(field)

        if _is_empty(off_val):
            offline[field] = llm_val              # LLM fills the blank
        else:
            if _normalize(off_val) != _normalize(llm_val):
                conflicts.append({"field": field, "offline": off_val, "llm": llm_val})
            # offline value stays — no overwrite


    # Review flag
    confidence = float(llm.get("_llm_confidence", 0.0))
    offline["_needs_review"]   = bool(conflicts) or confidence < 0.85 or "_error" in llm
    offline["_llm_confidence"] = confidence
    offline["_conflicts"]      = conflicts

    if conflicts:
        print(f"      ⚡ {len(conflicts)} conflict(s): {[c['field'] for c in conflicts]}")

    return offline


# ═══════════════════════════════════════════════════════════════════════
# TOP-LEVEL BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_meta(
    pdf_path:     Path,
    doc_id:       Optional[str] = None,
    use_crossref: bool = True,
    use_entrez:   bool = True,
    use_llm:      bool = True,
) -> dict:
    pdf_path = Path(pdf_path)
    if doc_id is None:
        doc_id = _get_doc_id(pdf_path)

    print(f"\n{'='*60}")
    print(f"PDF   : {pdf_path.name}")
    print(f"doc_id: {doc_id}")

    # ── State-aware idempotency check ─────────────────────────────
    # If a verified meta.json already exists locally, treat it as ground
    # truth and skip all extraction phases entirely.
    # If an unverified draft exists, use it as the starting base so that
    # any manually-edited fields are preserved across re-runs.
    existing_path = (EXTRACTED_DIR / doc_id / "meta.json")
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get("_needs_review") is False:
            print(f"  [✓] Verified local ground truth found. Skipping extraction.")
            return existing
        else:
            print(f"  [~] Unverified draft found. Using as base, re-running pipeline.")
            meta = existing   # preserve any manually-edited fields
    else:
        meta = _blank_meta(doc_id)

    # ── Apply offline ground truth (highest priority) ─────────────
    truth = OFFLINE_GROUND_TRUTH.get(doc_id, {})
    for k, v in truth.items():
        if k in meta:
            meta[k] = _enforce_type(k, v)
    if truth:
        print(f"  [GT] Ground truth: {list(truth.keys())}")


    # ── Phase 1a: PDF text ────────────────────────────────────────
    print("  [0] PDF text extraction...")
    try:
        pdf_text = extract_pdf_text(pdf_path, max_pages=4)
        print(f"      {len(pdf_text):,} chars")
    except Exception as e:
        print(f"      ERROR: {e}")
        return meta

    meta["doi"]               = meta.get("doi") or _find_doi(pdf_text)
    meta["guideline_version"] = meta.get("guideline_version") or _find_nccn_version(pdf_text)
    print(f"      DOI: {meta['doi'] or 'not found'}")

    abstract: Optional[str] = None

    # ── Phase 1b: CrossRef ────────────────────────────────────────
    if use_crossref and meta["doi"]:
        print("  [1] CrossRef...")
        cr = fetch_crossref_meta(meta["doi"])
        for f in ("title", "journal", "pub_year", "pub_month", "doi"):
            if _is_empty(meta.get(f)) and cr.get(f):
                meta[f] = cr[f]
        print(f"      title: {(meta['title'] or 'n/a')[:65]}")
        print(f"      year : {meta['pub_year']} / {meta['pub_month']}")
        time.sleep(0.5)
    elif not meta["doi"]:
        print("  [1] CrossRef skipped — no DOI")

    # ── Phase 1c: PubMed ──────────────────────────────────────────
    if use_entrez and meta["doi"]:
        print("  [2] PubMed...")
        pmid = search_pubmed_by_doi(meta["doi"])
        if pmid:
            print(f"      PMID: {pmid}")
            pm = fetch_pubmed_meta(pmid)
            for f in ("title", "journal", "pub_year", "pub_month", "doi"):
                if _is_empty(meta.get(f)) and pm.get(f):
                    meta[f] = pm[f]
            abstract = pm.get("abstract")
            print(f"      abstract: {len(abstract or '')} chars | pub_types: {pm.get('pub_types', [])[:3]}")

            if _is_empty(meta.get("cancer_type")):
                meta["cancer_type"] = _infer_cancer_type(pm.get("mesh_terms", [])) or "Breast"
            if _is_empty(meta.get("source_type")):
                hint = _infer_source_type(pm.get("pub_types", []))
                if hint:
                    meta["source_type"] = hint
                    print(f"      source_type hint: {hint}")
        else:
            print("      Not in PubMed")
        time.sleep(0.4)
    elif not meta["doi"]:
        print("  [2] Entrez skipped — no DOI")


    # ── Phase 2: LLM ──────────────────────────────────────────────
    if use_llm:
        backend  = API_CFG.get("llm_backend", "groq")
        groq_key = API_CFG.get("groq", {}).get("api_key", "")
        if backend == "groq" and groq_key in ("", "YOUR_GROQ_API_KEY_HERE"):
            print("  [3] LLM skipped — add key to .secrets/secrets.yaml")
        else:
            model = API_CFG.get(backend, {}).get("model", "")
            print(f"  [3] LLM ({backend}/{model})...")
            llm_raw = run_llm_extraction(pdf_text, abstract)
            if "_error" in llm_raw:
                print(f"      ERROR: {llm_raw['_error']}")
                meta["_needs_review"]   = True
                meta["_llm_confidence"] = 0.0
            else:
                print(f"      confidence: {llm_raw.get('_llm_confidence', 0):.2f}")
                meta = defensive_merge(meta, llm_raw)   # Phase 3
                print(f"      status: {'✓ clean' if not meta['_needs_review'] else '⚠ review'}")

    missing = [f for f in ("title", "source_type", "population", "line_of_therapy")
               if _is_empty(meta.get(f))]
    if missing:
        print(f"  ⚠  Still empty after all phases: {', '.join(missing)}")

    return meta


def save_meta(meta: dict) -> Path:
    doc_id   = meta["doc_id"]
    out_dir  = EXTRACTED_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meta.json"
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ✓ {out_path}")
    return out_path


def build_all(use_crossref=True, use_entrez=True, use_llm=True):
    raw_dir = RAW_DIR if RAW_DIR.is_absolute() else Path(__file__).parent.parent / RAW_DIR
    pdfs    = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {raw_dir}")
        return

    backend = API_CFG.get("llm_backend", "groq")
    print(f"Found {len(pdfs)} PDFs | CrossRef={use_crossref} Entrez={use_entrez} LLM={use_llm} ({backend})")

    results = []
    for pdf in pdfs:
        doc_id = _get_doc_id(pdf)
        meta   = build_meta(pdf, doc_id=doc_id, use_crossref=use_crossref,
                            use_entrez=use_entrez, use_llm=use_llm)
        save_meta(meta)
        results.append(meta)

    clean      = [m for m in results if not m["_needs_review"]]
    conflicted = [m for m in results if m.get("_conflicts")]
    print(f"\n{'='*60}")
    print(f"✓ {len(clean)}/{len(results)} clean | ⚡ {len(conflicted)} with offline↔LLM conflicts\n")
    for m in results:
        icon = "✓" if not m["_needs_review"] else "⚠"
        conf = m.get("_llm_confidence")
        print(f"  {icon} {m['doc_id']}  conf={conf:.2f}")
        for f in ("source_type", "cancer_subtype", "drug_focus", "line_of_therapy"):
            print(f"      {f:<20}: {str(m.get(f, '—'))[:55]}")
        if m.get("_conflicts"):
            print(f"      _conflicts          : {[c['field'] for c in m['_conflicts']]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="meta_builder v2 — defensive clinical metadata")
    p.add_argument("--pdf",          help="Single PDF path")
    p.add_argument("--doc-id",       help="Override doc_id")
    p.add_argument("--no-crossref",  action="store_true")
    p.add_argument("--no-entrez",    action="store_true")
    p.add_argument("--no-llm",       action="store_true")
    args = p.parse_args()

    opts = dict(use_crossref=not args.no_crossref,
                use_entrez=not args.no_entrez,
                use_llm=not args.no_llm)

    if args.pdf:
        save_meta(build_meta(Path(args.pdf), doc_id=args.doc_id, **opts))
    else:
        build_all(**opts)
