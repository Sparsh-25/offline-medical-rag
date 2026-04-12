# Oncology RAG Metadata Pipeline (Phase 0)

> A stateful, defensive data ingestion pipeline for building highly accurate Retrieval-Augmented Generation (RAG) systems in the medical domain.

## ⚠️ The Problem: Why Standard RAG Fails in Healthcare
Standard RAG pipelines dump raw PDF text into a vector database. In oncology, this is dangerous. If a user asks for "first-line treatments for HER2-low breast cancer," a naive vector search might retrieve a 10-year-old retrospective study instead of the 2024 ASCO Guideline, simply because of keyword overlap.

To solve this, we must tag every text chunk with **Clinical Semantic Metadata** (e.g., `cancer_subtype`, `drug_focus`, `source_type`). 
* **The Catch:** Public APIs (PubMed, CrossRef) do not provide granular clinical metadata.
* **The Danger:** Large Language Models (LLMs) can extract this data from text, but they are prone to hallucinations (e.g., misclassifying a database review as a Randomized Controlled Trial).

## 💡 The Solution: "Trust Boundaries" Architecture
This pipeline solves the metadata extraction problem using a **3-Phase Defensive Architecture**. It isolates deterministic factual data from probabilistic AI predictions, ensuring 0% hallucination in structural metadata.

### Phase 1: Deterministic Anchors (The "Robot")
Before the AI is invoked, the pipeline establishes undeniable bibliographic facts.
1. **PDF Parsing (`pypdfium2`):** Extracts text safely, handling ligatures and complex encodings. Regex isolates the DOI.
2. **CrossRef & PubMed APIs:** Uses the DOI to fetch the official Title, Journal, Publication Year, and Abstract.
* **Why:** Bibliographic data must be 100% accurate. Relying on APIs establishes a factual baseline that the LLM is not allowed to touch.

### Phase 2: Semantic Extraction Engine (The "Intern")
The pipeline passes the verified Abstract and first-page text to an LLM (via Groq API or local Ollama).
* The LLM is restricted by a strict **System Prompt** enforcing clinical translation rules (e.g., translating "estrogen receptor" to `["HR+"]`).
* It extracts arrays for `cancer_subtype`, `drug_focus`, and classifies the `source_type` (Guideline, RCT, Mechanism, etc.).
* Output is forced into a strict JSON schema.

### Phase 3: Defensive Merge & HITL Bootstrapping (The "Supervisor")
The pipeline attempts to merge Phase 1 (Facts) with Phase 2 (AI Draft).
* **Rule of Law:** The LLM is only permitted to fill `null` values. It is strictly forbidden from overwriting data acquired in Phase 1.
* **Conflict Resolution:** If a conflict occurs, the deterministic data wins, the conflict is logged in a `_conflicts` array, and the document is flagged (`_needs_review: true`).
* **Human-in-the-Loop (HITL):** If the pipeline relies heavily on the LLM (a "Cold Start" for a new paper), it flags the JSON for human review. A human quickly audits the JSON and changes `_needs_review: false`.

---

## 🧠 Stateful Idempotency (The Database Loop)
This pipeline acts as its own database. It is fully **state-aware** and **idempotent**.

When you run `python src/meta_builder.py`:
1. The script scans the `data/extracted/` directory.
2. If it finds a `meta.json` where `_needs_review` is `false`, it recognizes that file as **Verified Ground Truth**.
3. It instantly loads the file and skips the API and LLM phases entirely.

**Why this matters:** * **Zero API Costs on Rebuilds:** You can rebuild your entire vector database in seconds for $0.
* **Bootstrapping:** You only manually review an AI's output *once*. Once saved, that JSON becomes an immutable fact that defends against future AI hallucinations.

---

## 🛠️ The Target Schema

json
{
  "doc_id": "monaleesa2_subanalysis_2018",
  "title": "Ribociclib plus letrozole versus letrozole alone in patients with de novo HR+, HER2− advanced breast cancer",
  "pub_year": 2018,
  "source_type": "rct",
  "cancer_type": "Breast",
  "cancer_subtype": ["HR+", "HER2-"],
  "line_of_therapy": "first_line",
  "drug_focus": ["Ribociclib", "Letrozole"],
  "_needs_review": false
}


### Oncology RAG Metadata Pipeline (Phase 0)

> A stateful, defensive data ingestion pipeline for building highly accurate Retrieval-Augmented Generation (RAG) systems in the medical domain.

### ⚠️ The Problem: Why Standard RAG Fails in Healthcare
Standard RAG pipelines dump raw PDF text into a vector database. In oncology, this is dangerous. If a user asks for "first-line treatments for HER2-low breast cancer," a naive vector search might retrieve a 10-year-old retrospective study instead of the 2024 ASCO Guideline, simply because of keyword overlap.

To solve this, we must tag every text chunk with **Clinical Semantic Metadata** (e.g., `cancer_subtype`, `drug_focus`, `source_type`). 
* **The Catch:** Public APIs (PubMed, CrossRef) do not provide granular clinical metadata.
* **The Danger:** Large Language Models (LLMs) can extract this data from text, but they are prone to hallucinations (e.g., misclassifying a database review as a Randomized Controlled Trial).

### 💡 The Solution: "Trust Boundaries" Architecture
This pipeline solves the metadata extraction problem using a **3-Phase Defensive Architecture**. It isolates deterministic factual data from probabilistic AI predictions, ensuring 0% hallucination in structural metadata.

### Phase 1: Deterministic Anchors (The "Robot")
Before the AI is invoked, the pipeline establishes undeniable bibliographic facts.
1. **PDF Parsing (`pypdfium2`):** Extracts text safely, handling ligatures and complex encodings. Regex isolates the DOI.
2. **CrossRef & PubMed APIs:** Uses the DOI to fetch the official Title, Journal, Publication Year, and Abstract.
* **Why:** Bibliographic data must be 100% accurate. Relying on APIs establishes a factual baseline that the LLM is not allowed to touch.

### Phase 2: Semantic Extraction Engine (The "Intern")
The pipeline passes the verified Abstract and first-page text to an LLM (via Groq API or local Ollama).
* The LLM is restricted by a strict **System Prompt** enforcing clinical translation rules (e.g., translating "estrogen receptor" to `["HR+"]`).
* It extracts arrays for `cancer_subtype`, `drug_focus`, and classifies the `source_type` (Guideline, RCT, Mechanism, etc.).
* Output is forced into a strict JSON schema.

### Phase 3: Defensive Merge & HITL Bootstrapping (The "Supervisor")
The pipeline attempts to merge Phase 1 (Facts) with Phase 2 (AI Draft).
* **Rule of Law:** The LLM is only permitted to fill `null` values. It is strictly forbidden from overwriting data acquired in Phase 1.
* **Conflict Resolution:** If a conflict occurs, the deterministic data wins, the conflict is logged in a `_conflicts` array, and the document is flagged (`_needs_review: true`).
* **Human-in-the-Loop (HITL):** If the pipeline relies heavily on the LLM (a "Cold Start" for a new paper), it flags the JSON for human review. A human quickly audits the JSON and changes `_needs_review: false`.

---

## 🧠 Stateful Idempotency (The Database Loop)
This pipeline acts as its own database. It is fully **state-aware** and **idempotent**.

When you run `python src/meta_builder.py`:
1. The script scans the `data/extracted/` directory.
2. If it finds a `meta.json` where `_needs_review` is `false`, it recognizes that file as **Verified Ground Truth**.
3. It instantly loads the file and skips the API and LLM phases entirely.

**Why this matters:** * **Zero API Costs on Rebuilds:** You can rebuild your entire vector database in seconds for $0.
* **Bootstrapping:** You only manually review an AI's output *once*. Once saved, that JSON becomes an immutable fact that defends against future AI hallucinations.

---

## 🛠️ The Target Schema

json
{
  "doc_id": "monaleesa2_subanalysis_2018",
  "title": "Ribociclib plus letrozole versus letrozole alone in patients with de novo HR+, HER2− advanced breast cancer",
  "pub_year": 2018,
  "source_type": "rct",
  "cancer_type": "Breast",
  "cancer_subtype": ["HR+", "HER2-"],
  "line_of_therapy": "first_line",
  "drug_focus": ["Ribociclib", "Letrozole"],
  "_needs_review": false
}
(Note: Schema enforces arrays for drugs and subtypes to handle combination therapies and multi-biomarker profiles.)

## 🚀 Setup & Usage

### 1. Environment Configuration
Create a `.secrets/secrets.yaml` file to store your API keys:
```yaml
api:
  entrez_email: "your.email@university.edu"
  llm_backend: "groq" # or "ollama" for local execution
  groq:
    api_key: "gsk_..."
    model: "llama-3.1-8b-instant"

### 2. File Structure
Place your raw PDFs in the `data/raw/` folder so the script can locate them.
```text
project_root/
├── data/
│   ├── raw/                 # Drop medical PDFs here
│   └── extracted/           # Pipeline generates folders & meta.json here
├── src/
│   └── meta_builder.py      # The pipeline script
└── .secrets/


3. Execution
Run the batch builder to process all new PDFs in your raw directory:

Bash
python src/meta_builder.py
To process a specific PDF entirely offline (skipping the LLM extraction):

Bash
python src/meta_builder.py --pdf data/raw/monaleesa2.pdf --no-llm

4. The Human Audit
After running the script, open your data/extracted/ directory. Any meta.json flagged with "_needs_review": true requires a quick visual audit. Correct any hallucinated clinical fields, set the flag to false, and save the file. Your metadata pipeline is now successfully bootstrapped and safely locked in.