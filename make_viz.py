"""Render the 4-depth classification tree and the flat-projection distribution.

All five strategic branches, including "Other feedback", come straight from
the taxonomy embedded in out/results.json; every count is a real assignment
row from out/flat.json. Writes PNG and SVG to out/viz/ and, when the marker
comments are present, injects the SVGs into NOTES.html so the notes stay a
single self-contained file.
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

REPO = Path(__file__).parent
OUT = REPO / "out" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

results = json.loads((REPO / "out" / "results.json").read_text("utf-8"))
flat = json.loads((REPO / "out" / "flat.json").read_text("utf-8"))
taxonomy = results["taxonomy"]

specific_counts = Counter(row["specific_theme"] for row in flat)
review_total = len(results["review_results"])
assigned_reviews = len({row["review_id"] for row in flat})
abstained = review_total - assigned_reviews
specific_total = sum(
    len(mid["specific_themes"])
    for strategic in taxonomy["strategic_themes"]
    for mid in strategic["midlevel_themes"]
)

palette = ["#176b87", "#b3541e", "#4f772d", "#7d4f9a", "#6b7280"]
ROOT_COLOR = "#17202a"

# render_structure: (strategic_label, color, [(mid, [(leaf, count)])])
render_structure = []
for s_idx, strategic in enumerate(taxonomy["strategic_themes"]):
    mids = []
    for mid in strategic["midlevel_themes"]:
        mids.append((
            mid["label"],
            [
                (leaf["label"], specific_counts.get(leaf["label"], 0))
                for leaf in mid["specific_themes"]
            ],
        ))
    render_structure.append(
        (strategic["label"], palette[s_idx % len(palette)], mids)
    )

# ---------------------------------------------------------------- tree image
leaf_total = sum(
    len(leaves) for _, _, mids in render_structure for _, leaves in mids
)
fig, ax = plt.subplots(figsize=(16.5, 0.44 * leaf_total + 2.6), dpi=150)
ax.set_xlim(0, 12)
ax.invert_yaxis()
ax.axis("off")

X_ROOT, X_STRAT, X_MID, X_LEAF = 0.1, 2.5, 5.6, 8.7
y = 0.0
GAP_LEAF, GAP_MID, GAP_STRAT = 1.0, 0.55, 0.9

def box(x, y, text, color, size, weight="normal", alpha=0.14):
    ax.text(
        x, y, text, fontsize=size, fontweight=weight, va="center", ha="left",
        color="#17202a",
        bbox=dict(
            boxstyle="round,pad=0.32", facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.3,
        ),
        zorder=3,
    )

strat_ys = []
for strat_label, color, mids in render_structure:
    mid_ys = []
    for mid_label, leaves in mids:
        leaf_ys = []
        for leaf_label, count in leaves:
            box(X_LEAF, y, f"{leaf_label}  ({count})", color, 9.5)
            leaf_ys.append(y)
            y += GAP_LEAF
        mid_y = sum(leaf_ys) / len(leaf_ys)
        mid_count = sum(count for _, count in leaves)
        box(X_MID, mid_y, f"{mid_label}  ({mid_count})", color, 10.5, "bold")
        for leaf_y in leaf_ys:
            ax.plot(
                [X_MID + 2.55, X_LEAF - 0.12], [mid_y, leaf_y],
                color=color, linewidth=0.9, alpha=0.55, zorder=1,
            )
        mid_ys.append(mid_y)
        y += GAP_MID
    strat_y = sum(mid_ys) / len(mid_ys)
    strat_count = sum(count for _, leaves in mids for _, count in leaves)
    box(
        X_STRAT, strat_y, f"{strat_label}  ({strat_count})",
        color, 12, "bold", alpha=0.22,
    )
    for mid_y in mid_ys:
        ax.plot(
            [X_STRAT + 2.85, X_MID - 0.12], [strat_y, mid_y],
            color=color, linewidth=1.2, alpha=0.55, zorder=1,
        )
    strat_ys.append((strat_y, color))
    y += GAP_STRAT

root_y = sum(sy for sy, _ in strat_ys) / len(strat_ys)
box(
    X_ROOT, root_y,
    f"All reviews\n({review_total} reviews,\n{len(flat)} assignments)",
    ROOT_COLOR, 11, "bold", alpha=0.10,
)
for strat_y, color in strat_ys:
    ax.plot(
        [X_ROOT + 1.9, X_STRAT - 0.12], [root_y, strat_y],
        color=color, linewidth=1.4, alpha=0.55, zorder=1,
    )

ax.set_title(
    "Classification tree - root / strategic / midlevel / specific. Counts are "
    f"assignment rows from {len(flat)} flat.json rows covering "
    f"{assigned_reviews} of {review_total} reviews"
    + (f"; {abstained} reviews abstained." if abstained else "."),
    fontsize=12, loc="left", pad=14,
)
fig.tight_layout()
fig.savefig(OUT / "taxonomy_tree.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "taxonomy_tree.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------- flat projection image
bars = []
for strat_label, color, mids in render_structure:
    entries = [
        (leaf_label, count)
        for _, leaves in mids
        for leaf_label, count in leaves
    ]
    entries.sort(key=lambda item: -item[1])
    for leaf_label, count in entries:
        bars.append((leaf_label, count, color, strat_label))

fig, ax = plt.subplots(figsize=(12, 13), dpi=150)
positions = range(len(bars))
ax.barh(
    list(positions),
    [count for _, count, _, _ in bars],
    color=[color for _, _, color, _ in bars],
    alpha=0.85, height=0.72,
)
ax.set_yticks(list(positions))
ax.set_yticklabels([label for label, _, _, _ in bars], fontsize=9)
ax.invert_yaxis()
for position, (_, count, _, _) in zip(positions, bars):
    ax.text(count + 0.25, position, str(count), va="center", fontsize=8.5)
seen = set()
handles = []
for _, _, color, strat_label in bars:
    if strat_label not in seen:
        seen.add(strat_label)
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, color=color, alpha=0.85, label=strat_label,
        ))
ax.legend(handles=handles, loc="lower right", fontsize=10, title="Strategic theme")
ax.set_xlabel("flat.json rows (multi-label assignments)")
ax.set_title(
    f"Flat projection - {len(flat)} rows over {specific_total} specific "
    f"themes ({assigned_reviews} of {review_total} reviews assigned)",
    fontsize=12, loc="left", pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "flat_projection.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "flat_projection.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote", OUT / "taxonomy_tree.png", "and .svg")
print("wrote", OUT / "flat_projection.png", "and .svg")

# ------------------------------------------- inject SVGs into NOTES.html
def _svg_body(path: Path) -> str:
    text = path.read_text("utf-8")
    start = text.index("<svg")
    svg = text[start:]
    # Scale with the page (viewBox + CSS) instead of the fixed matplotlib size.
    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)
    return svg

notes_path = REPO / "NOTES.html"
notes = notes_path.read_text("utf-8")
injected = 0
for marker, svg_file in (
    ("viz-tree", "taxonomy_tree.svg"),
    ("viz-projection", "flat_projection.svg"),
):
    start_marker = f"<!-- {marker}:start -->"
    end_marker = f"<!-- {marker}:end -->"
    if start_marker in notes and end_marker in notes:
        head, rest = notes.split(start_marker, 1)
        _, tail = rest.split(end_marker, 1)
        notes = (
            head + start_marker + "\n" + _svg_body(OUT / svg_file)
            + "\n" + end_marker + tail
        )
        injected += 1
if injected:
    notes_path.write_text(notes, "utf-8")
    print(f"injected {injected} SVG(s) into NOTES.html")
else:
    print("NOTES.html markers not found; skipped injection")
