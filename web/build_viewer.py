"""
build_viewer.py — build a standalone, offline demo viewer from the answer bank.

Reads eval/demo_bank.jsonl (produced by eval/run_demo_bank.py on the box) and writes a
single self-contained web/demo_viewer.html with the data inlined. No server, no backend,
no GPU: open the HTML file directly in a browser and click through the questions, answers,
sources, and timing.

We inline the data (rather than fetch it) on purpose: a browser opening a file:// page
cannot fetch a local .jsonl. Rebuild whenever the bank changes.

    python -m web.build_viewer
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = _ROOT / "eval" / "demo_bank.jsonl"
OUT_PATH = Path(__file__).parent / "demo_viewer.html"

# The page shell. __DATA__ is replaced with the bank as a JSON array. The CSS mirrors
# web/static/index.html so the live app and the offline viewer look like one system.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oncology RAG — demo viewer</title>
<style>
  :root { --bg:#f7f7f8; --card:#fff; --line:#e3e3e8; --ink:#1a1a1f; --muted:#6b6b75;
          --accent:#2f6f4f; --good:#2f6f4f; --bad:#b0342c; --warn:#8a6d00; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:18px 16px 10px; max-width:1100px; margin:0 auto; }
  h1 { font-size:19px; margin:0 0 4px; }
  .stats { color:var(--muted); font-size:13px; }
  .wrap { max-width:1100px; margin:0 auto; padding:8px 16px 60px; display:grid; grid-template-columns:360px 1fr; gap:18px; }
  @media (max-width:760px){ .wrap{ grid-template-columns:1fr; } }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 12px; }
  .chip { border:1px solid var(--line); background:var(--card); border-radius:999px; padding:4px 11px; font-size:13px; cursor:pointer; }
  .chip.on { background:var(--ink); color:#fff; border-color:var(--ink); }
  .list { display:flex; flex-direction:column; gap:7px; max-height:76vh; overflow:auto; }
  .q { border:1px solid var(--line); background:var(--card); border-radius:9px; padding:9px 11px; cursor:pointer; }
  .q.sel { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
  .q .top { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); margin-bottom:3px; }
  .q .txt { font-size:13px; }
  .tag { font-size:11px; padding:1px 7px; border-radius:999px; font-weight:600; }
  .tag.answer { background:#e7f1ec; color:var(--good); }
  .tag.abstain { background:#fdeeec; color:var(--bad); }
  .dot { margin-left:auto; font-size:12px; font-weight:700; }
  .dot.ok { color:var(--good); } .dot.miss { color:var(--warn); }
  .detail { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  .detail h2 { font-size:15px; margin:0 0 8px; }
  .badges { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 12px; font-size:12px; }
  .badge { border:1px solid var(--line); border-radius:7px; padding:3px 9px; color:var(--muted); }
  .answer { white-space:pre-wrap; }
  .answer.abstained { color:var(--bad); font-style:italic; }
  .srch2 { font-size:12px; letter-spacing:.03em; text-transform:uppercase; color:var(--muted); margin:18px 0 8px; }
  .src { border:1px solid var(--line); border-radius:8px; padding:9px 11px; margin-top:9px; }
  .src .hd { font-size:12px; color:var(--accent); font-weight:600; }
  .src .snip { font-size:12.5px; color:#333; margin-top:5px; max-height:7em; overflow:auto; white-space:pre-wrap; }
  .empty { color:var(--muted); padding:20px 4px; }
</style>
</head>
<body>
<header>
  <h1>Offline Oncology RAG — demo viewer</h1>
  <div class="stats" id="stats"></div>
</header>
<div class="wrap">
  <div>
    <div class="chips" id="chips"></div>
    <div class="list" id="list"></div>
  </div>
  <div class="detail" id="detail"><div class="empty">Select a question on the left.</div></div>
</div>

<script>
const DATA = __DATA__;

function isCorrect(r){ return r.expected === (r.abstained ? "abstain" : "answer"); }
function esc(s){ return (s||"").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function median(a){ if(!a.length) return 0; const s=[...a].sort((x,y)=>x-y); const m=s.length>>1;
  return s.length%2 ? s[m] : (s[m-1]+s[m])/2; }

const FILTERS = [
  ["all","All"], ["answer","Answerable"], ["abstain","Abstain"], ["miss","Mismatches"]
];
let active = "all";
let selected = null;

function matches(r){
  if(active==="all") return true;
  if(active==="miss") return !isCorrect(r);
  return r.expected === active;
}

function renderStats(){
  const ans = DATA.filter(r=>r.expected==="answer");
  const abs = DATA.filter(r=>r.expected==="abstain");
  const ansOk = ans.filter(r=>!r.abstained).length;
  const absOk = abs.filter(r=>r.abstained).length;
  const med = median(DATA.map(r=>r.timing && r.timing.total_s).filter(x=>x!=null));
  document.getElementById("stats").textContent =
    `${DATA.length} questions · answerable ${ansOk}/${ans.length} answered · abstain ${absOk}/${abs.length} abstained · median ${med.toFixed(1)}s end-to-end`;
}

function renderChips(){
  const el = document.getElementById("chips"); el.innerHTML = "";
  for(const [k,label] of FILTERS){
    const c = document.createElement("div");
    c.className = "chip" + (active===k?" on":"");
    c.textContent = label + (k!=="all" ? ` (${DATA.filter(r=> k==="miss"?!isCorrect(r):r.expected===k).length})` : "");
    c.onclick = () => { active=k; render(); };
    el.appendChild(c);
  }
}

function renderList(){
  const el = document.getElementById("list"); el.innerHTML = "";
  const rows = DATA.filter(matches);
  if(!rows.length){ el.innerHTML = '<div class="empty">None.</div>'; return; }
  for(const r of rows){
    const d = document.createElement("div");
    d.className = "q" + (selected===r.id?" sel":"");
    const ok = isCorrect(r);
    d.innerHTML =
      `<div class="top"><span class="tag ${r.expected}">${r.expected}</span>`+
      `<span>${esc(r.category)}</span>`+
      `<span class="dot ${ok?'ok':'miss'}">${ok?'✓':'⚠'}</span></div>`+
      `<div class="txt">${esc(r.question)}</div>`;
    d.onclick = () => { selected=r.id; render(); };
    el.appendChild(d);
  }
}

function renderDetail(){
  const el = document.getElementById("detail");
  const r = DATA.find(x=>x.id===selected);
  if(!r){ el.innerHTML = '<div class="empty">Select a question on the left.</div>'; return; }
  const ok = isCorrect(r);
  const t = r.timing || {};
  let html = `<h2>${esc(r.question)}</h2>`;
  html += `<div class="badges">`+
    `<span class="badge">expected: ${r.expected}</span>`+
    `<span class="badge">actual: ${r.abstained?"abstained":"answered"}</span>`+
    `<span class="badge" style="color:${ok?'var(--good)':'var(--warn)'}">${ok?'as expected':'mismatch'}</span>`+
    `<span class="badge">${esc(r.category)} · ${esc(r.origin)}</span>`+
    `<span class="badge">ret ${t.retrieval_s ?? '?'}s · gen ${t.generation_s ?? '?'}s · total ${t.total_s ?? '?'}s</span>`+
    `</div>`;
  html += `<div class="answer${r.abstained?' abstained':''}">${esc(r.answer)}</div>`;
  if(r.sources && r.sources.length){
    html += `<div class="srch2">Sources the answer was grounded in</div>`;
    for(const s of r.sources){
      html += `<div class="src"><div class="hd">[${s.rank}] ${esc(s.doc_id)} · §${esc(s.section||"")}</div>`+
              `<div class="snip">${esc(s.snippet||"")}</div></div>`;
    }
  }
  el.innerHTML = html;
}

function render(){ renderChips(); renderList(); renderDetail(); }
renderStats(); render();
</script>
</body>
</html>
"""


def main() -> None:
    if not BANK_PATH.exists():
        raise SystemExit(f"No bank at {BANK_PATH}. Run `python -m eval.run_demo_bank` on the box first.")
    rows = [json.loads(l) for l in BANK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    # json.dumps is safe to drop into a <script> here because our data has no "</script>";
    # guard anyway by escaping any stray closing tag.
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    OUT_PATH.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"{len(rows)} questions -> {OUT_PATH}")
    print("Open that file directly in a browser (no server, no GPU needed).")


if __name__ == "__main__":
    main()
