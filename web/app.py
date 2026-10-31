"""
web/app.py — a minimal Flask app to test the live RAG pipeline in a browser.

This is NOT a production server. It is the smallest thing that lets us type a question
in a browser and watch the real pipeline answer it end to end: retrieve -> rerank ->
safety-net -> Qwen3-32B -> grounded answer + sources. It exists to validate that the
whole serving path works while we have GPU access.

Where it runs: ON THE GPU BOX (that is where the LLM and the index live). The box has no
SSH, only JupyterLab, so the browser reaches this app through Jupyter's port proxy — see
the run instructions in web/README or the chat. One Flask process serves both the HTML
page and the /ask API, so there is no CORS or cross-origin setup.

    python -m web.app          # from the repo root, inside the box venv

Then open the proxied URL (e.g. https://<jupyter-host>/proxy/8000/) in your browser.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory

from src.answer import answer, load_llm, load_reranker_once

_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load((_ROOT / "config.yaml").read_text())

# Chunk texts, loaded once, so we can show the source passage behind each citation.
_CHUNK_TEXTS: dict[str, str] = {}


def _load_chunk_texts() -> None:
    path = _ROOT / _CFG["paths"]["chunks"]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            _CHUNK_TEXTS[row["chunk_id"]] = row.get("chunk_text", "")


app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def home():
    """Serve the single-page UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Run one question through the real pipeline and return the answer + its sources."""
    query = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    started = time.time()
    result = answer(query)   # retrieve -> rerank -> safety-net -> Qwen3-32B (config.yaml)

    # Attach the source passage behind each citation so the UI can show what grounded it.
    for source in result["sources"]:
        source["snippet"] = _CHUNK_TEXTS.get(source["chunk_id"], "")[:600]
    result["seconds"] = round(time.time() - started, 1)
    return jsonify(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the live RAG test UI.")
    # 8000 is often already taken on the box; pick any free port and reach it the same way.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8050)))
    args = parser.parse_args()

    _load_chunk_texts()
    # Warm the heavy models once at startup so the first browser request isn't a cold load.
    print("Warming models (Qwen3-32B + reranker) — the first load is slow ...", flush=True)
    load_reranker_once()
    load_llm()
    print(f"Ready. Serving on port {args.port}.", flush=True)
    # threaded=False: the 32B model answers one request at a time anyway, and this keeps
    # the single shared llama-cpp context from being hit concurrently.
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
