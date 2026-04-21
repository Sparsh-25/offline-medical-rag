"""
caption.py — Ollama/LLaVA figure captioning + clinical extraction pipeline
============================================================================
Handles ALL oncology figure types:
  Kaplan-Meier, Forest Plot, Bar/Line/Box/Scatter/Waterfall Charts,
  Heatmaps, Tables, Flowcharts, CONSORT Diagrams, Biological Pathways,
  Medical Imaging (IHC/Scans), Pie Charts, and Unknown types.

Output per figure → RAG-ready markdown:
  ![{rich clinical summary}](figures/filename.png)
  <!-- FIGURE_METADATA: {structured JSON} -->
  {enriched table / flowchart / pathway text if applicable}
"""

import json
import re
import time
import yaml
import requests
import base64
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal, Dict, Any

_ROOT = Path(__file__).parent.parent
cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))

# ── VLM backend config (config.yaml → vlm section, sensible defaults) ────────
_VLM_CFG     = cfg.get("vlm", {})
VLM_MODEL    = _VLM_CFG.get("model",    "llava")
VLM_TIMEOUT  = int(_VLM_CFG.get("timeout",  300))
VLM_BASE_URL = _VLM_CFG.get("base_url", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VLM PROMPT  (universal — type-specific rules embedded)
# ─────────────────────────────────────────────────────────────────────────────

VLM_PROMPT = (
    "You are a Clinical Oncology Figure Extraction AI. "
    "Analyze the figure image and its surrounding text context to produce "
    "a comprehensive, structured JSON extraction.\n\n"

    "Figure ID: {figure_id}\n"
    "Context (caption / surrounding paragraph): \"{docling_extracted_context}\"\n\n"

    "══ STEP 1 — IDENTIFY ARCHETYPES ═══════════════════════════════════════\n"
    "Choose ALL archetypes that apply (use exact names):\n"
    "  Kaplan-Meier | Forest Plot | Bar Chart | Line Graph | Box Plot\n"
    "  Scatter Plot | Heatmap | Table | Flowchart | CONSORT Diagram\n"
    "  Biological Pathway | Medical Imaging | Pie Chart | Waterfall Plot | Other\n\n"

    "══ STEP 2 — UNIVERSAL RULES ════════════════════════════════════════════\n"
    "- Use BOTH image and text. If values conflict, ALWAYS prefer text.\n"
    "- NEVER infer or calculate. Return null for any missing value.\n"
    "- For every numeric metric return raw_value (exact string) AND numeric_value (float or null).\n"
    "- If a value is visually estimated from axis/graph, set is_estimated: true.\n"
    "- Extract ALL visible labels, legends, footnotes, and annotations.\n\n"

    "══ STEP 3 — TYPE-SPECIFIC EXTRACTION RULES ═════════════════════════════\n\n"

    "[KAPLAN-MEIER / SURVIVAL CURVES]\n"
    "→ cohorts: name, N, role (treatment|control|overall)\n"
    "→ hazard_ratio with 95% CI and p_value (log-rank)\n"
    "→ data_points: median_OS / median_PFS / median_DFS per cohort with CI\n"
    "→ data_points: at-risk counts at landmark timepoints\n"
    "→ x_axis: time unit (months/years), y_axis: probability or % survival\n\n"

    "[FOREST PLOTS]\n"
    "→ data_points: each subgroup row — metric=subgroup label, value=HR[95%CI]\n"
    "→ hazard_ratio: overall pooled estimate, confidence_interval: overall 95%CI\n"
    "→ p_value: overall. Add heterogeneity stats (I², Q, p-het) as data_points\n\n"

    "[BAR / COLUMN CHARTS]\n"
    "→ data_points: each bar — metric=category, value=numeric, unit=y-axis units\n"
    "→ Include error bars in value string e.g. '45.2 ± 3.1' or '45.2 (42.0-48.4)'\n"
    "→ statistical_annotations: significance markers between groups e.g. '* group_A vs group_B'\n\n"

    "[LINE GRAPHS]\n"
    "→ data_points: key inflection/endpoint values per series — metric=series+timepoint, value=y\n"
    "→ x_axis + y_axis labels and units\n"
    "→ trend direction (increasing/decreasing/plateau) in summary\n\n"

    "[BOX / VIOLIN PLOTS]\n"
    "→ data_points per group: metric=group, value=median (Q1–Q3), unit if available\n"
    "→ statistical_annotations: pairwise comparisons e.g. 'p=0.03: groupA vs groupB'\n\n"

    "[SCATTER PLOTS]\n"
    "→ data_points: correlation coefficient R or R², trend equation if shown\n"
    "→ x_axis + y_axis labels, units, and value ranges\n"
    "→ key clusters or outlier groups in summary\n\n"

    "[HEATMAPS]\n"
    "→ table_headers: column labels (up to 10)\n"
    "→ table_rows: up to 10 most informative rows with cell values\n"
    "→ data_points: top 5 extreme cells — metric='row_col', value=cell_value\n"
    "→ color scale meaning in summary\n\n"

    "[TABLES]\n"
    "→ table_headers: ALL column header strings\n"
    "→ table_rows: ALL rows — row_label + values dict {column: cell_value}\n"
    "→ data_points: 3-5 most clinically important cells as a summary highlight\n"
    "→ Include footnote markers (*, †, ‡) in the cell values\n\n"

    "[FLOWCHARTS / CONSORT / TRIAL DESIGN DIAGRAMS]\n"
    "→ flow_steps: EVERY box sequentially — label, count (int or null), step_type\n"
    "  step_type options: enrollment | screened | excluded | randomized | allocated\n"
    "                     treated | follow-up | analyzed | lost | discontinued\n"
    "→ cohorts: each randomized arm with N\n"
    "→ data_points: total_enrolled, total_randomized, total_analyzed per arm\n\n"

    "[BIOLOGICAL PATHWAYS / MECHANISM DIAGRAMS]\n"
    "→ key_entities: ALL named receptors, ligands, kinases, TFs, downstream targets\n"
    "→ drug_targets: any drug names or targets highlighted\n"
    "→ pathway_relations: source → target with interaction\n"
    "  interaction options: activates | inhibits | phosphorylates | upregulates\n"
    "                       downregulates | binds | cleaves | recruits | ubiquitinates\n"
    "→ Oncogenic vs tumor-suppressor context in summary\n\n"

    "[MEDICAL IMAGING — IHC / SCANS]\n"
    "→ data_points: biomarker, staining_intensity (0–3+ or H-score), clinical_classification\n"
    "→ key_entities: tissue type, stain/modality name, treatment timepoint\n"
    "→ scale bar value if shown in summary\n\n"

    "══ STEP 4 — OUTPUT FORMAT ══════════════════════════════════════════════\n"
    "Return ONLY the JSON below. No markdown fences, no explanations.\n"
    "summary MUST be 3-4 sentences: (a) what the figure shows, (b) key numeric finding, "
    "(c) clinical implication.\n"
    "clinical_relevance: one concise sentence on why this finding matters.\n\n"
    "{\n"
    '  "figure_id": "string",\n'
    '  "archetypes": ["string"],\n'
    '  "figure_type_confidence": "high|medium|low",\n'
    '  "title": "string or null",\n'
    '  "x_axis": {"label": "string or null", "unit": "string or null"},\n'
    '  "y_axis": {"label": "string or null", "unit": "string or null"},\n'
    '  "cohorts": [{"name": "string", "role": "treatment|control|overall|unknown", "n": integer_or_null}],\n'
    '  "hazard_ratio": {"raw_value": "string", "numeric_value": float_or_null, "confidence": "high|medium|low", "is_estimated": false},\n'
    '  "p_value": {"raw_value": "string", "numeric_value": float_or_null, "confidence": "high|medium|low", "is_estimated": false},\n'
    '  "confidence_interval": {"raw_value": "string", "lower": float_or_null, "upper": float_or_null, "confidence_level": 0.95},\n'
    '  "data_points": [{"metric": "string", "value": "string", "unit": "string or null", "cohort": "string or null"}],\n'
    '  "table_headers": ["string"],\n'
    '  "table_rows": [{"row_label": "string", "values": {"column_name": "cell_value"}}],\n'
    '  "flow_steps": [{"label": "string", "count": integer_or_null, "step_type": "string or null"}],\n'
    '  "key_entities": ["string"],\n'
    '  "drug_targets": ["string"],\n'
    '  "pathway_relations": [{"source": "string", "target": "string", "interaction": "string or null"}],\n'
    '  "statistical_annotations": ["string"],\n'
    '  "summary": "string",\n'
    '  "clinical_relevance": "string or null"\n'
    "}\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PYDANTIC SCHEMA — full universal schema for all figure types
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedMetric(BaseModel):
    raw_value:     Optional[str]   = None
    numeric_value: Optional[float] = None
    confidence:    str             = "high"
    is_estimated:  bool            = False

class ConfidenceInterval(BaseModel):
    raw_value:        Optional[str]   = None
    lower:            Optional[float] = None
    upper:            Optional[float] = None
    confidence_level: float           = 0.95

class AxisInfo(BaseModel):
    label: Optional[str] = None
    unit:  Optional[str] = None

class Cohort(BaseModel):
    name: str
    role: Literal["treatment", "control", "overall", "unknown"] = "unknown"
    n:    Optional[int] = None

class ClinicalDataPoint(BaseModel):
    metric: str
    value:  Optional[str] = None
    unit:   Optional[str] = None
    cohort: Optional[str] = None

class TableRow(BaseModel):
    row_label: str
    values:    Dict[str, str] = Field(default_factory=dict)

class FlowStep(BaseModel):
    label:     str
    count:     Optional[int] = None
    step_type: Optional[str] = None

class PathwayRelation(BaseModel):
    source:      str
    target:      str
    interaction: Optional[str] = None

class VisualExtraction(BaseModel):
    figure_id:              str
    archetypes:             List[str]             = Field(default_factory=list)
    figure_type_confidence: str                   = "high"
    title:                  Optional[str]          = None
    x_axis:                 Optional[AxisInfo]     = None
    y_axis:                 Optional[AxisInfo]     = None
    cohorts:                List[Cohort]           = Field(default_factory=list)
    hazard_ratio:           Optional[ExtractedMetric]     = None
    p_value:                Optional[ExtractedMetric]     = None
    confidence_interval:    Optional[ConfidenceInterval]  = None
    data_points:            List[ClinicalDataPoint]       = Field(default_factory=list)
    table_headers:          List[str]             = Field(default_factory=list)
    table_rows:             List[TableRow]        = Field(default_factory=list)
    flow_steps:             List[FlowStep]        = Field(default_factory=list)
    key_entities:           List[str]             = Field(default_factory=list)
    drug_targets:           List[str]             = Field(default_factory=list)
    pathway_relations:      List[PathwayRelation] = Field(default_factory=list)
    statistical_annotations:List[str]             = Field(default_factory=list)
    summary:                str                   = ""
    clinical_relevance:     Optional[str]          = None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAFE VALIDATORS  — post-Pydantic numeric corrections
# ─────────────────────────────────────────────────────────────────────────────

def safe_validate_hr(metric: Optional[ExtractedMetric]) -> Optional[ExtractedMetric]:
    if metric and metric.numeric_value is not None:
        if metric.numeric_value <= 0:
            metric.numeric_value = None
            metric.confidence = "low"
    return metric

def safe_validate_p_value(metric: Optional[ExtractedMetric]) -> Optional[ExtractedMetric]:
    if metric and metric.numeric_value is not None:
        if not (0.0 <= metric.numeric_value <= 1.0):
            metric.numeric_value = None
            metric.confidence = "low"
    return metric

def safe_validate_ci(ci: Optional[ConfidenceInterval]) -> Optional[ConfidenceInterval]:
    if ci and ci.raw_value:
        try:
            clean = (
                ci.raw_value
                  .replace("\u2013", " ").replace("\u2014", " ")
                  .replace("-", " ").replace("to", " ").replace(",", ".")
            )
            nums = re.findall(r"(\d*\.?\d+)", clean)
            if len(nums) >= 2:
                lo, hi = float(nums[-2]), float(nums[-1])
                if lo < hi:
                    ci.lower, ci.upper = lo, hi
                else:
                    ci.lower = ci.upper = None
            else:
                ci.lower = ci.upper = None
        except Exception:
            ci.lower = ci.upper = None
    return ci

def apply_safe_validations(extraction: VisualExtraction) -> VisualExtraction:
    extraction.hazard_ratio      = safe_validate_hr(extraction.hazard_ratio)
    extraction.p_value           = safe_validate_p_value(extraction.p_value)
    extraction.confidence_interval = safe_validate_ci(extraction.confidence_interval)
    return extraction


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METADATA BUILDER  — DB-friendly dict for structured RAG filtering
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(extraction: VisualExtraction) -> dict:
    archetypes_lower = {a.lower() for a in extraction.archetypes}

    metadata: dict = {
        "figure_id":               extraction.figure_id,
        "archetypes":              extraction.archetypes,
        "figure_type_confidence":  extraction.figure_type_confidence,
        "title":                   extraction.title,
        "cohorts":                 [c.name for c in extraction.cohorts],
        "cohort_sizes":            {c.name: c.n for c in extraction.cohorts if c.n},
        "metrics":                 list({dp.metric for dp in extraction.data_points if dp.metric}),
        "key_entities":            extraction.key_entities,
        "drug_targets":            extraction.drug_targets,
        "has_statistical_tests":   len(extraction.statistical_annotations) > 0,
        # booleans for fast filter
        "is_survival_curve":  bool(archetypes_lower & {"kaplan-meier", "km", "survival curve"}),
        "is_forest_plot":     "forest plot" in archetypes_lower,
        "is_table":           "table" in archetypes_lower,
        "is_flowchart":       bool(archetypes_lower & {"flowchart", "consort diagram", "consort", "trial design"}),
        "is_pathway":         bool(archetypes_lower & {"biological pathway", "pathway", "mechanism"}),
        "is_imaging":         "medical imaging" in archetypes_lower,
    }

    # ── Hazard ratio ─────────────────────────────────────────────
    if extraction.hazard_ratio and extraction.hazard_ratio.numeric_value is not None:
        hr = extraction.hazard_ratio.numeric_value
        metadata["hazard_ratio"] = hr
        metadata["has_hr"]       = True
        metadata["hr_benefit"]   = hr < 1.0
    else:
        metadata["has_hr"] = False

    # ── P-value ──────────────────────────────────────────────────
    if extraction.p_value and extraction.p_value.numeric_value is not None:
        p = extraction.p_value.numeric_value
        metadata["p_value"]       = p
        metadata["is_significant"] = p < 0.05
    else:
        metadata["is_significant"] = False

    # ── Confidence interval ──────────────────────────────────────
    if extraction.confidence_interval and extraction.confidence_interval.lower is not None:
        ci = extraction.confidence_interval
        metadata["ci_range"] = [ci.lower, ci.upper]
        metadata["ci_width"] = round(ci.upper - ci.lower, 4)
        if metadata.get("has_hr"):
            metadata["is_statistically_significant"] = (
                (ci.lower > 1.0) or (ci.upper < 1.0)
            )

    # ── Table dimensions ────────────────────────────────────────
    if extraction.table_rows:
        metadata["table_row_count"]    = len(extraction.table_rows)
        metadata["table_column_count"] = len(extraction.table_headers)

    # ── Flowchart patient counts ─────────────────────────────────
    if extraction.flow_steps:
        for step in extraction.flow_steps:
            if step.step_type == "enrollment" and step.count:
                metadata["total_enrolled"] = step.count
            if step.step_type == "randomized" and step.count:
                metadata["total_randomized"] = step.count
            if step.step_type == "analyzed" and step.count:
                metadata.setdefault("total_analyzed", step.count)

    # ── Pathway richness ────────────────────────────────────────
    if extraction.pathway_relations:
        metadata["pathway_relation_count"] = len(extraction.pathway_relations)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXTRACTION STATUS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def get_extraction_status(extraction: VisualExtraction) -> str:
    archetypes_lower = {a.lower() for a in extraction.archetypes}

    # Archetype category flags
    is_quantitative = bool(archetypes_lower & {"kaplan-meier", "km", "survival curve", "forest plot"})
    is_table        = "table" in archetypes_lower
    is_flowchart    = bool(archetypes_lower & {"flowchart", "consort", "consort diagram", "trial design"})
    is_pathway      = bool(archetypes_lower & {"biological pathway", "pathway", "mechanism"})

    # Data presence flags
    has_hr       = extraction.hazard_ratio and extraction.hazard_ratio.numeric_value is not None
    has_p        = extraction.p_value and extraction.p_value.numeric_value is not None
    has_points   = len(extraction.data_points) > 0
    has_table    = len(extraction.table_rows) > 0 or len(extraction.table_headers) > 0
    has_flow     = len(extraction.flow_steps) > 0
    has_pathway  = len(extraction.key_entities) > 0 or len(extraction.pathway_relations) > 0
    has_summary  = bool(extraction.summary.strip())

    if is_quantitative:
        if has_hr and has_p:
            return "success"
        if has_hr or has_p or has_points:
            return "partial"
        return "failed"

    if is_table:
        return "success" if has_table  else ("partial" if (has_points or has_summary) else "failed")

    if is_flowchart:
        return "success" if has_flow   else ("partial" if (has_points or has_summary) else "failed")

    if is_pathway:
        return "success" if has_pathway else ("partial" if has_summary else "failed")

    # Generic: bar, line, box, scatter, heatmap, imaging, unknown
    if has_points or has_summary:
        return "success"
    return "failed"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  JSON HEALER  — strips LLM conversational noise, isolates JSON object
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
# 7.  INPUT SANITISER  — cleans LLaVA type-coercion problems pre-Pydantic
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_llava_output(data: dict, figure_id: str) -> dict:
    """Coerce raw LLaVA JSON into types Pydantic expects. Never raises."""

    data["figure_id"] = figure_id          # always override — never trust LLM

    # ── Simple list fields → default to [] ──────────────────────
    for lf in ("archetypes", "cohorts", "data_points", "table_headers",
                "table_rows", "flow_steps", "key_entities", "drug_targets",
                "pathway_relations", "statistical_annotations"):
        if not isinstance(data.get(lf), list):
            data[lf] = []

    # ── Dict-or-None fields → default to None ───────────────────
    for df in ("hazard_ratio", "p_value", "confidence_interval", "x_axis", "y_axis"):
        if not isinstance(data.get(df), (dict, type(None))):
            data[df] = None

    # ── String fields ────────────────────────────────────────────
    for sf in ("summary", "title", "figure_type_confidence", "clinical_relevance"):
        v = data.get(sf)
        if v is not None and not isinstance(v, str):
            data[sf] = str(v)
    if not isinstance(data.get("figure_type_confidence"), str):
        data["figure_type_confidence"] = "high"

    # ── Cohorts — must be list of dicts with valid 'role' ────────
    valid_roles = {"treatment", "control", "overall", "unknown"}
    clean_cohorts = []
    for c in data["cohorts"]:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        if c.get("role") not in valid_roles:
            c["role"] = "unknown"
        if not isinstance(c.get("n"), (int, type(None))):
            try:
                c["n"] = int(float(str(c["n"]).replace(",", "")))
            except (ValueError, TypeError):
                c["n"] = None
        clean_cohorts.append(c)
    data["cohorts"] = clean_cohorts

    # ── data_points — must have 'metric'; coerce value/unit/cohort to str ──
    clean_dps = []
    for dp in data["data_points"]:
        if not isinstance(dp, dict) or not dp.get("metric"):
            continue
        for str_field in ("value", "unit", "cohort"):
            v = dp.get(str_field)
            if v is not None and not isinstance(v, str):
                dp[str_field] = str(v)
        clean_dps.append(dp)
    data["data_points"] = clean_dps

    # ── table_rows — must have 'row_label' and 'values' dict ────
    # LLaVA sometimes returns cell values as int/float/None — coerce all to str.
    clean_rows = []
    for row in data["table_rows"]:
        if not isinstance(row, dict):
            continue
        row.setdefault("row_label", "unknown")
        vals = row.get("values")
        if not isinstance(vals, dict):
            row["values"] = {}
        else:
            row["values"] = {
                k: (str(v) if v is not None else "") for k, v in vals.items()
            }
        clean_rows.append(row)
    data["table_rows"] = clean_rows

    # ── flow_steps — must have 'label' ──────────────────────────
    clean_steps = []
    for step in data["flow_steps"]:
        if not isinstance(step, dict) or not step.get("label"):
            continue
        if not isinstance(step.get("count"), (int, type(None))):
            try:
                step["count"] = int(float(str(step["count"]).replace(",", "")))
            except (ValueError, TypeError):
                step["count"] = None
        clean_steps.append(step)
    data["flow_steps"] = clean_steps

    # ── pathway_relations — must have source + target ────────────
    data["pathway_relations"] = [
        r for r in data["pathway_relations"]
        if isinstance(r, dict) and r.get("source") and r.get("target")
    ]

    # ── List[str] fields — coerce every item to str, drop None ───
    # LLaVA sometimes returns dicts, ints, or nested lists inside these.
    for list_str_field in ("statistical_annotations", "drug_targets", "key_entities"):
        raw = data.get(list_str_field, [])
        if not isinstance(raw, list):
            data[list_str_field] = []
        else:
            data[list_str_field] = [
                str(item) for item in raw if item is not None
            ]

    # ── ExtractedMetric sub-objects — coerce str/float fields ────
    # hazard_ratio / p_value have Optional[str] fields (confidence, method,
    # raw_value) that LLaVA occasionally returns as numbers or booleans.
    for metric_field in ("hazard_ratio", "p_value"):
        m = data.get(metric_field)
        if isinstance(m, dict):
            for sf in ("raw_value", "confidence", "method"):
                v = m.get(sf)
                if v is not None and not isinstance(v, str):
                    m[sf] = str(v)
            nv = m.get("numeric_value")
            if nv is not None and not isinstance(nv, (int, float)):
                try:
                    m["numeric_value"] = float(
                        str(nv).replace(",", "").strip()
                    )
                except (ValueError, TypeError):
                    m["numeric_value"] = None

    # ── ConfidenceInterval — confidence_level must be float ──────
    # LLaVA often returns "95%" or "0.95" instead of 95.0.
    ci = data.get("confidence_interval")
    if isinstance(ci, dict):
        cl = ci.get("confidence_level")
        if cl is not None and not isinstance(cl, (int, float)):
            try:
                cl_float = float(str(cl).strip().rstrip("%").strip())
                if 0 < cl_float <= 1.0:   # 0.95 → 95.0
                    cl_float *= 100
                ci["confidence_level"] = cl_float
            except (ValueError, TypeError):
                ci["confidence_level"] = None
        rv = ci.get("raw_value")
        if rv is not None and not isinstance(rv, str):
            ci["raw_value"] = str(rv)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OLLAMA SERVER CHECK
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
# 9.  CORE VLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(
    server_online: bool,
    img_path: Path,
    figure_id: str,
    docling_context: str,
) -> Optional[VisualExtraction]:
    """
    Send one figure to LLaVA via Ollama and return a validated VisualExtraction.
    Returns None on any failure (never raises).
    """
    try:
        if not server_online:
            return None

        with open(img_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")

        # Use .replace() — NOT .format() — because the prompt contains JSON schema
        # with curly braces like {column_name}, {label} etc. that .format() mistakes
        # for format variables and raises KeyError.
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
            print(f"    [WARN] LLaVA returned non-dict JSON for {img_path.name}")
            return None

        data = _sanitise_llava_output(data, figure_id)

        try:
            extraction = VisualExtraction(**data)
        except ValidationError as ve:
            print(f"    [DEBUG] ValidationError ({ve.error_count()} errors) for {img_path.name}")
            for err in ve.errors()[:5]:
                print(f"      {err['loc']} → {err['msg']}")
            return None

        extraction = apply_safe_validations(extraction)
        return extraction

    except Exception as e:
        print(f"    Extraction failed for {img_path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENRICHED TEXT BUILDER  — extra searchable text for tables / flowcharts / pathways
# ─────────────────────────────────────────────────────────────────────────────

def build_enriched_text(extraction_data: dict) -> str:
    """
    Convert structured extraction data into plain-text / markdown that the
    RAG chunker can index and retrieve.  Only emitted for figure types that
    carry tabular or sequential information (tables, flowcharts, pathways).
    """
    parts: list[str] = []
    archetypes_lower = {a.lower() for a in extraction_data.get("archetypes", [])}

    # ── Extracted Table → markdown table ─────────────────────────────────────
    headers   = extraction_data.get("table_headers", [])
    table_rows = extraction_data.get("table_rows", [])
    if headers and table_rows:
        parts.append("")
        parts.append("| " + " | ".join(str(h) for h in headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table_rows[:20]:           # cap at 20 rows
            cells = [str(row.get("values", {}).get(h, "")) for h in headers]
            parts.append("| " + " | ".join(cells) + " |")

    # ── Flowchart / CONSORT → numbered list ──────────────────────────────────
    flow_steps = extraction_data.get("flow_steps", [])
    is_flowchart_doc = bool(archetypes_lower & {
        "flowchart", "consort", "consort diagram", "trial design"
    })
    if flow_steps and is_flowchart_doc:
        parts.append("")
        parts.append("**Trial / Study Flow:**")
        for step in flow_steps:
            count_str = f" (n={step['count']:,})" if step.get("count") else ""
            type_str  = f" [{step['step_type']}]"  if step.get("step_type") else ""
            parts.append(f"- {step['label']}{count_str}{type_str}")

    # ── Pathway relations → entity interaction list ───────────────────────────
    relations  = extraction_data.get("pathway_relations", [])
    entities   = extraction_data.get("key_entities", [])
    drug_targets = extraction_data.get("drug_targets", [])
    if relations or entities:
        parts.append("")
        if drug_targets:
            parts.append(f"**Drug Targets:** {', '.join(drug_targets)}")
        if entities:
            parts.append(f"**Key Entities:** {', '.join(entities[:15])}")
        if relations:
            parts.append("**Key Interactions:**")
            for rel in relations[:15]:
                interaction = rel.get("interaction") or "→"
                parts.append(f"- {rel['source']} {interaction} {rel['target']}")

    # ── Statistical annotations ───────────────────────────────────────────────
    annotations = extraction_data.get("statistical_annotations", [])
    if annotations:
        parts.append("")
        parts.append("**Statistical Comparisons:** " + "; ".join(annotations[:8]))

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 11. PLACEHOLDER FORMAT DETECTION
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
# 12. REPLACEMENT MAP BUILDER
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

    fig_dir         = doc_dir / "figures"
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

            print(f"({elapsed:.1f}s) archetype={extraction.archetypes} status={status}")

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
                "metadata":        {"extraction_status": "failed"},
                "filename":        fig_info["filename"],
                "extraction_data": None,
            }
            fig_info["extraction_status"] = "failed"

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return replacement_map


# ─────────────────────────────────────────────────────────────────────────────
# 13. MARKDOWN INJECTOR
# ─────────────────────────────────────────────────────────────────────────────

def inject_captions_into_markdown(
    md_text:          str,
    replacement_map:  dict[str, Any],
    fig_dir_relative: str = "figures",
) -> tuple[str, int]:
    """
    Replace every figure placeholder in the markdown with:

      ![{VLM clinical summary}](figures/{filename}.png)
      <!-- FIGURE_METADATA: {structured JSON} -->
      {enriched table / flow / pathway text if applicable}

    Handles all 5 placeholder formats and is idempotent on re-runs.
    """
    replacements_made = 0

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

    def _pop_item_by_ref(ref: Optional[str]) -> Optional[dict]:
        """Consume and return a replacement_map entry by self_ref or sequentially."""
        if ref:
            item = replacement_map.get(ref)
            if item is not None:
                replacement_map[ref] = None
                return item
        # Sequential fallback
        for k in list(replacement_map.keys()):
            if replacement_map[k] is not None:
                item = replacement_map[k]
                replacement_map[k] = None
                return item
        return None

    def _pop_item_by_filename(filename: str) -> Optional[dict]:
        """Consume and return a replacement_map entry matching filename."""
        # Exact filename match first
        for k in list(replacement_map.keys()):
            item = replacement_map[k]
            if item and item.get("filename") == filename:
                replacement_map[k] = None
                return item
        # Sequential fallback
        for k in list(replacement_map.keys()):
            if replacement_map[k] is not None:
                item = replacement_map[k]
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

        # Append enriched text for tables, flowcharts, pathways
        ext_data = item.get("extraction_data")
        if ext_data:
            enriched = build_enriched_text(ext_data)
            if enriched:
                output += enriched

        replacements_made += 1
        return output

    # ── Format 1: <!-- image ref: #/pictures/N --> ────────────────────────────
    def _replace_html_comment(m: re.Match) -> str:
        raw      = m.group(0)
        ref_match = re.search(r"ref:\s*([^\s,>]+)", raw)
        ref      = ref_match.group(1) if ref_match else None
        item     = _pop_item_by_ref(ref)
        return _render(item) if item else raw

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
    def _replace_fallback_alt(m: re.Match) -> str:
        fname = Path(m.group(2)).name
        item  = _pop_item_by_filename(fname)
        return _render(item, default_filename=fname) if item else m.group(0)

    md_text = re.sub(
        r"!\[Figure from oncology document[^\]]*\]\(([^)]+)\)",
        lambda m: _replace_fallback_alt(
            # re-wrap so group(2) = path (group(1) = alt-text)
            type("_M", (), {"group": lambda self, i: [None, m.group(1), m.group(1)][i]})()
        ) if False else m.group(0),   # placeholder — use guarded regex below
        md_text,
        flags=re.IGNORECASE,
    )
    # Correct approach — use a two-group regex
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
# 14. DOCUMENT-LEVEL ORCHESTRATION
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

    md_text = md_path.read_text(encoding="utf-8")
    fmt     = find_placeholder_pattern(md_text)
    print(f"  Placeholder format: {fmt}")

    updated_md, n_replaced = inject_captions_into_markdown(
        md_text, replacement_map, fig_dir_relative="figures"
    )
    md_path.write_text(updated_md, encoding="utf-8")

    # Count only entries that had actual content (not action=remove entries which
    # are for skipped/tiny figures that never had a real placeholder in the markdown).
    real_entries = sum(
        1 for v in replacement_map.values()
        if v is not None and v.get("action") != "remove"
    )
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
            if ext_data:
                enriched = build_enriched_text(ext_data)
                if enriched:
                    block += enriched
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


def caption_all():
    ext_dir  = Path(cfg["paths"]["extracted"])
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
    caption_all()