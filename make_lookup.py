"""Build lookup.html: a self-contained, filterable review-assignment browser.

Reads the frozen taxonomy, the source reviews, and the rich classification
output, and writes a single HTML file with no external dependencies. Rerun
after any new classification run to refresh the page:

    python make_lookup.py
    python make_lookup.py --results out/results.json --output lookup.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review and assignment lookup</title>
<style>
  :root {
    --ink: #17202a;
    --muted: #6b7a8c;
    --line: #dde4ec;
    --paper: #f5f7fa;
    --card: #ffffff;
    --accent: #1f6feb;
    --accent-soft: #e7f0fe;
    --mark: #fff1a8;
    --chip1: #eef2ff;
    --chip2: #f0fdf4;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    background: var(--paper);
    line-height: 1.5;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--line);
    padding: 20px 28px 14px;
  }
  header h1 { margin: 0 0 2px; font-size: 21px; }
  header .sub { color: var(--muted); font-size: 13px; }
  .toolbar {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--card);
    border-bottom: 1px solid var(--line);
    padding: 12px 28px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    align-items: center;
  }
  .group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .group > label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-right: 2px;
  }
  .pill {
    border: 1px solid var(--line);
    background: var(--card);
    color: var(--ink);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 13px;
    cursor: pointer;
  }
  .pill:hover { border-color: var(--accent); }
  .pill.on {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  select, input[type="search"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 5px 8px;
    font-size: 13px;
    background: var(--card);
    color: var(--ink);
    max-width: 300px;
  }
  input[type="search"] { width: 230px; }
  .count-line {
    padding: 10px 28px 0;
    color: var(--muted);
    font-size: 13px;
  }
  .count-line b { color: var(--ink); }
  a.reset { color: var(--accent); cursor: pointer; margin-left: 10px; }
  main {
    padding: 14px 28px 60px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
    gap: 14px;
    align-items: start;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
  }
  .card .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
    align-items: baseline;
    font-size: 12px;
    color: var(--muted);
  }
  .stars { color: #e8a20c; letter-spacing: 1px; font-size: 13px; }
  .card h3 { margin: 6px 0 4px; font-size: 15px; }
  .card .content { font-size: 14px; }
  mark { background: var(--mark); padding: 0 1px; border-radius: 2px; }
  details.original { margin-top: 6px; font-size: 12.5px; color: var(--muted); }
  details.original summary { cursor: pointer; }
  .assignments { margin-top: 10px; display: flex; flex-direction: column; gap: 8px; }
  .assignment {
    border-left: 3px solid var(--accent);
    background: var(--accent-soft);
    border-radius: 0 8px 8px 0;
    padding: 7px 10px;
    font-size: 13px;
  }
  .path { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
  .path .sep { color: var(--muted); }
  .tag {
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 12px;
    background: var(--chip1);
    border: 1px solid #d6defb;
    cursor: pointer;
    white-space: nowrap;
  }
  .tag.leaf { background: var(--chip2); border-color: #c8ecd2; font-weight: 600; }
  .tag:hover { border-color: var(--accent); }
  .evidence { margin-top: 4px; color: #40506b; font-style: italic; }
  .none-note {
    margin-top: 10px;
    font-size: 13px;
    color: var(--muted);
    border-left: 3px solid var(--line);
    padding: 4px 10px;
  }
  .badge {
    font-size: 11px;
    border-radius: 999px;
    padding: 1px 8px;
    border: 1px solid var(--line);
    color: var(--muted);
  }
  .empty {
    grid-column: 1 / -1;
    text-align: center;
    color: var(--muted);
    padding: 60px 0;
  }
  footer {
    padding: 0 28px 40px;
    color: var(--muted);
    font-size: 12px;
  }
</style>
</head>
<body>
<header>
  <h1>Review and assignment lookup</h1>
  <div class="sub" id="run-line"></div>
</header>

<div class="toolbar">
  <div class="group" id="count-filters">
    <label>Assignments</label>
    <button class="pill on" data-count="all">All</button>
    <button class="pill" data-count="none">None</button>
    <button class="pill" data-count="single">Single</button>
    <button class="pill" data-count="multiple">Multiple</button>
  </div>
  <div class="group">
    <label>Tree</label>
    <select id="sel-strategic"><option value="">All strategic themes</option></select>
    <select id="sel-midlevel"><option value="">All mid-level themes</option></select>
    <select id="sel-specific"><option value="">All specific themes</option></select>
  </div>
  <div class="group" id="rating-filters">
    <label>Rating</label>
    <button class="pill on" data-rating="all">All</button>
    <button class="pill" data-rating="1">1</button>
    <button class="pill" data-rating="2">2</button>
    <button class="pill" data-rating="3">3</button>
    <button class="pill" data-rating="4">4</button>
    <button class="pill" data-rating="5">5</button>
  </div>
  <div class="group">
    <input type="search" id="search" placeholder="Search text, evidence, themes&#8230;">
  </div>
</div>

<div class="count-line" id="count-line"></div>
<main id="cards"></main>
<footer id="footer"></footer>

<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);

// ---- taxonomy lookups ------------------------------------------------------
const leafInfo = {};   // leaf id -> {leaf, mid, strat}
const midInfo = {};    // mid id -> {mid, strat}
for (const strat of DATA.taxonomy.strategic_themes) {
  for (const mid of strat.midlevel_themes) {
    midInfo[mid.id] = { mid, strat };
    for (const leaf of mid.specific_themes) {
      leafInfo[leaf.id] = { leaf, mid, strat };
    }
  }
}

// per-node assignment counts (assignments, not reviews)
const leafCounts = {}, midCounts = {}, stratCounts = {};
for (const review of DATA.reviews) {
  for (const a of review.assignments) {
    const info = leafInfo[a.specific_theme_id];
    if (!info) continue;
    leafCounts[info.leaf.id] = (leafCounts[info.leaf.id] || 0) + 1;
    midCounts[info.mid.id] = (midCounts[info.mid.id] || 0) + 1;
    stratCounts[info.strat.id] = (stratCounts[info.strat.id] || 0) + 1;
  }
}

// ---- state -----------------------------------------------------------------
const state = {
  count: "all",       // all | none | single | multiple
  strategic: "",
  midlevel: "",
  specific: "",
  rating: "all",
  query: "",
};

// ---- filter dropdowns ------------------------------------------------------
const selStrategic = document.getElementById("sel-strategic");
const selMidlevel = document.getElementById("sel-midlevel");
const selSpecific = document.getElementById("sel-specific");

function option(value, label) {
  const el = document.createElement("option");
  el.value = value;
  el.textContent = label;
  return el;
}

function fillStrategic() {
  for (const strat of DATA.taxonomy.strategic_themes) {
    selStrategic.appendChild(
      option(strat.id, `${strat.label} (${stratCounts[strat.id] || 0})`)
    );
  }
}

function fillMidlevel() {
  selMidlevel.length = 1;
  const strats = DATA.taxonomy.strategic_themes.filter(
    (s) => !state.strategic || s.id === state.strategic
  );
  for (const strat of strats) {
    for (const mid of strat.midlevel_themes) {
      selMidlevel.appendChild(
        option(mid.id, `${mid.label} (${midCounts[mid.id] || 0})`)
      );
    }
  }
  selMidlevel.value = state.midlevel;
}

function fillSpecific() {
  selSpecific.length = 1;
  for (const [id, info] of Object.entries(leafInfo)) {
    if (state.midlevel && info.mid.id !== state.midlevel) continue;
    if (!state.midlevel && state.strategic && info.strat.id !== state.strategic) continue;
    selSpecific.appendChild(
      option(id, `${info.leaf.label} (${leafCounts[id] || 0})`)
    );
  }
  selSpecific.value = state.specific;
}

selStrategic.addEventListener("change", () => {
  state.strategic = selStrategic.value;
  state.midlevel = "";
  state.specific = "";
  fillMidlevel();
  fillSpecific();
  render();
});
selMidlevel.addEventListener("change", () => {
  state.midlevel = selMidlevel.value;
  if (state.midlevel) {
    state.strategic = midInfo[state.midlevel].strat.id;
    selStrategic.value = state.strategic;
  }
  state.specific = "";
  fillSpecific();
  render();
});
selSpecific.addEventListener("change", () => {
  state.specific = selSpecific.value;
  if (state.specific) {
    const info = leafInfo[state.specific];
    state.midlevel = info.mid.id;
    state.strategic = info.strat.id;
    selStrategic.value = state.strategic;
    fillMidlevel();
    selMidlevel.value = state.midlevel;
    fillSpecific();
    selSpecific.value = state.specific;
  }
  render();
});

// ---- pill groups -----------------------------------------------------------
function wirePills(containerId, attribute, stateKey) {
  const container = document.getElementById(containerId);
  container.addEventListener("click", (event) => {
    const pill = event.target.closest(".pill");
    if (!pill) return;
    state[stateKey] = pill.dataset[attribute];
    for (const el of container.querySelectorAll(".pill")) {
      el.classList.toggle("on", el === pill);
    }
    render();
  });
}
wirePills("count-filters", "count", "count");
wirePills("rating-filters", "rating", "rating");

let searchTimer = null;
document.getElementById("search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = event.target.value.trim().toLowerCase();
    render();
  }, 120);
});

function setLeafFilter(leafId) {
  state.specific = leafId;
  const info = leafInfo[leafId];
  state.midlevel = info.mid.id;
  state.strategic = info.strat.id;
  selStrategic.value = state.strategic;
  fillMidlevel();
  fillSpecific();
  window.scrollTo({ top: 0, behavior: "smooth" });
  render();
}

function resetFilters() {
  state.count = "all";
  state.strategic = "";
  state.midlevel = "";
  state.specific = "";
  state.rating = "all";
  state.query = "";
  document.getElementById("search").value = "";
  selStrategic.value = "";
  fillMidlevel();
  fillSpecific();
  for (const [id, key] of [["count-filters", "count"], ["rating-filters", "rating"]]) {
    for (const el of document.getElementById(id).querySelectorAll(".pill")) {
      el.classList.toggle("on", el.dataset[key] === "all");
    }
  }
  render();
}

// ---- filtering -------------------------------------------------------------
function matchesTree(review) {
  if (!state.strategic && !state.midlevel && !state.specific) return true;
  return review.assignments.some((a) => {
    const info = leafInfo[a.specific_theme_id];
    if (!info) return false;
    if (state.specific) return info.leaf.id === state.specific;
    if (state.midlevel) return info.mid.id === state.midlevel;
    return info.strat.id === state.strategic;
  });
}

function matchesCount(review) {
  const n = review.assignments.length;
  if (state.count === "none") return n === 0;
  if (state.count === "single") return n === 1;
  if (state.count === "multiple") return n >= 2;
  return true;
}

function matchesQuery(review) {
  if (!state.query) return true;
  const q = state.query;
  if (review.id.toLowerCase().includes(q)) return true;
  if ((review.title || "").toLowerCase().includes(q)) return true;
  if (review.content_en.toLowerCase().includes(q)) return true;
  if ((review.content_no || "").toLowerCase().includes(q)) return true;
  return review.assignments.some((a) => {
    const info = leafInfo[a.specific_theme_id];
    return (
      a.evidence.toLowerCase().includes(q) ||
      a.specific_theme_id.includes(q) ||
      (info &&
        (info.leaf.label.toLowerCase().includes(q) ||
          info.mid.label.toLowerCase().includes(q) ||
          info.strat.label.toLowerCase().includes(q)))
    );
  });
}

// ---- rendering -------------------------------------------------------------
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlight(contentEn, assignments) {
  // Mark each evidence span in the escaped English text. Evidence strings
  // are verbatim substrings by contract, so plain indexOf is sufficient;
  // overlapping spans are merged.
  const spans = [];
  for (const a of assignments) {
    const start = contentEn.indexOf(a.evidence);
    if (start >= 0) spans.push([start, start + a.evidence.length]);
  }
  spans.sort((x, y) => x[0] - y[0]);
  const merged = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
    else merged.push([...span]);
  }
  let html = "", cursor = 0;
  for (const [start, end] of merged) {
    html += escapeHtml(contentEn.slice(cursor, start));
    html += "<mark>" + escapeHtml(contentEn.slice(start, end)) + "</mark>";
    cursor = end;
  }
  return html + escapeHtml(contentEn.slice(cursor));
}

function card(review) {
  const stars = "\u2605".repeat(review.rating) + "\u2606".repeat(5 - review.rating);
  const date = (review.feedback_date || "").slice(0, 10);
  const n = review.assignments.length;
  const badge = n === 0 ? "no assignments" : n === 1 ? "1 assignment" : `${n} assignments`;

  let assignmentsHtml = "";
  if (n === 0) {
    assignmentsHtml =
      '<div class="none-note">No theme assigned' +
      (review.no_assignment_reason
        ? " &mdash; " + escapeHtml(review.no_assignment_reason)
        : "") +
      "</div>";
  } else {
    assignmentsHtml =
      '<div class="assignments">' +
      review.assignments
        .map((a) => {
          const info = leafInfo[a.specific_theme_id];
          const path = info
            ? `<span class="tag" data-leaf="${info.leaf.id}">${escapeHtml(info.strat.label)}</span>` +
              '<span class="sep">&rsaquo;</span>' +
              `<span class="tag" data-leaf="${info.leaf.id}">${escapeHtml(info.mid.label)}</span>` +
              '<span class="sep">&rsaquo;</span>' +
              `<span class="tag leaf" data-leaf="${info.leaf.id}" title="${escapeHtml(info.leaf.definition)}">${escapeHtml(info.leaf.label)}</span>`
            : `<span class="tag leaf">${escapeHtml(a.specific_theme_id)}</span>`;
          return (
            '<div class="assignment"><div class="path">' + path + "</div>" +
            '<div class="evidence">&ldquo;' + escapeHtml(a.evidence) + "&rdquo;</div></div>"
          );
        })
        .join("") +
      "</div>";
  }

  const original =
    review.content_no && review.content_no !== review.content_en
      ? '<details class="original"><summary>Original text</summary>' +
        escapeHtml(review.content_no) + "</details>"
      : "";

  return (
    '<article class="card">' +
    '<div class="meta">' +
    `<span class="stars" title="rating ${review.rating}/5">${stars}</span>` +
    `<span>${escapeHtml(review.id)}</span>` +
    `<span>${escapeHtml(date)}</span>` +
    `<span class="badge">${badge}</span>` +
    "</div>" +
    (review.title ? `<h3>${escapeHtml(review.title)}</h3>` : "") +
    `<div class="content">${highlight(review.content_en, review.assignments)}</div>` +
    original +
    assignmentsHtml +
    "</article>"
  );
}

const cardsEl = document.getElementById("cards");
cardsEl.addEventListener("click", (event) => {
  const tag = event.target.closest(".tag[data-leaf]");
  if (tag) setLeafFilter(tag.dataset.leaf);
});

function render() {
  const visible = DATA.reviews.filter(
    (r) =>
      matchesCount(r) &&
      matchesTree(r) &&
      (state.rating === "all" || r.rating === Number(state.rating)) &&
      matchesQuery(r)
  );
  const assignments = visible.reduce((sum, r) => sum + r.assignments.length, 0);
  document.getElementById("count-line").innerHTML =
    `Showing <b>${visible.length}</b> of ${DATA.reviews.length} reviews ` +
    `(<b>${assignments}</b> assignments)` +
    '<a class="reset" id="reset">Reset filters</a>';
  document.getElementById("reset").addEventListener("click", resetFilters);
  cardsEl.innerHTML = visible.length
    ? visible.map(card).join("")
    : '<div class="empty">No reviews match the current filters.</div>';
}

// ---- header ----------------------------------------------------------------
(function init() {
  const run = DATA.run;
  const total = DATA.reviews.reduce((s, r) => s + r.assignments.length, 0);
  document.getElementById("run-line").textContent =
    `${DATA.reviews.length} reviews, ${total} assignments \u00b7 ` +
    `model ${run.model} \u00b7 prompt ${DATA.prompt_version || "n/a"} \u00b7 ` +
    `run of ${run.created_at ? run.created_at.slice(0, 10) : "n/a"}`;
  document.getElementById("footer").textContent =
    `Generated ${DATA.generated_at} from out/results.json, themes.json, and ` +
    "data/reviews.json by make_lookup.py. Highlighted spans are the verbatim " +
    "evidence substrings the classifier cited; click any theme tag to filter by it.";
  fillStrategic();
  fillMidlevel();
  fillSpecific();
  render();
})();
</script>
</body>
</html>
"""


def build_payload(
    reviews_path: Path,
    taxonomy_path: Path,
    results_path: Path,
) -> dict:
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    by_id = {entry["review_id"]: entry for entry in results["review_results"]}
    merged = []
    for review in reviews:
        result = by_id.get(review["id"])
        if result is None:
            continue  # subset runs classify only part of the corpus
        merged.append(
            {
                "id": review["id"],
                "rating": review["rating"],
                "title": review.get("title") or "",
                "feedback_date": review.get("feedback_date") or "",
                "content_en": review["content_en"],
                "content_no": review.get("content_no") or "",
                "assignments": result["assignments"],
                "no_assignment_reason": result.get("no_assignment_reason"),
            }
        )
    from feedback_themes.pipeline import PROMPT_VERSION

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run": results.get("run", {}),
        "prompt_version": PROMPT_VERSION,
        "taxonomy": results.get("taxonomy") or taxonomy,
        "reviews": merged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", default=ROOT / "data" / "reviews.json", type=Path)
    parser.add_argument("--taxonomy", default=ROOT / "themes.json", type=Path)
    parser.add_argument("--results", default=ROOT / "out" / "results.json", type=Path)
    parser.add_argument("--output", default=ROOT / "lookup.html", type=Path)
    args = parser.parse_args()

    payload = build_payload(args.reviews, args.taxonomy, args.results)
    # "</" must not appear inside an inline <script> block.
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    args.output.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    print(
        f"wrote {args.output} ({len(payload['reviews'])} reviews, "
        f"{sum(len(r['assignments']) for r in payload['reviews'])} assignments)"
    )


if __name__ == "__main__":
    main()
