"""Review app for the product records and the frozen v1 manifests.

Browse products by split, category, label and label decision; compare the full and redacted
text next to the photo; and work through the hand-check queue of sampled contradictions
between the description and the catalog label. Verdicts are saved per reviewer to
annotations/conflict_check.csv, so Claude's verdicts and yours can be compared. Claude's
verdict on an item stays hidden until you have saved your own, and nothing is pre-selected,
so your verdicts are independent of Claude's. Test splits never appear.

Run from the repo root:
    uv run streamlit run app/review.py
"""

import csv
import html
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from pac import ontology as ontology_module
from pac.records import load_products
from pac.redact import redact_description, redact_title
from pac.review import ANNOTATION_FIELDS, REVIEW_SPLITS, VERDICTS, hand_check_queue
from pac.splits import TEST_SPLITS, matching_text

ROOT = Path(__file__).resolve().parents[1]
STYLE_DIRS = [ROOT / "data" / "raw" / "kaggle" / "styles", ROOT / "data" / "raw" / "myntra_mirror" / "styles"]
MANIFEST_DIR = ROOT / "manifests" / "v1"
IMAGE_DIR = ROOT / "data" / "images"
QUEUE_FILE = ROOT / "annotations" / "hand_check_queue_v0.csv"
ANNOTATIONS = Path(os.environ.get("PAC_ANNOTATIONS_PATH", ROOT / "annotations" / "conflict_check.csv"))
HUMAN = "human"
ATTRIBUTE_NAMES = {"neckline": "Neckline", "sleeve_length": "Sleeve length"}
STATUS_NAMES = {"agrees": "names the label", "conflicts": "names another label", "silent": "names no label"}
SPOT_CHECK_PER_VERDICT = 10
QUEUE_VIEWS = ["Not yet reviewed by you", "Spot check: 20 of Claude's calls", "Claude can't tell",
               "You and Claude disagree", "All 100"]
STYLE = """
<style>
mark.agree { background: rgba(46, 160, 67, 0.28); color: inherit; padding: 0 2px; border-radius: 3px; }
mark.other { background: rgba(218, 54, 51, 0.28); color: inherit; padding: 0 2px; border-radius: 3px; }
.text-block { line-height: 1.65; max-width: 78ch; }
</style>
"""


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_resource
def ontology():
    return ontology_module.load()


@st.cache_resource(show_spinner="Loading product records…")
def products_by_id(styles_dir: str) -> dict:
    return {p.id: p for p in load_products(styles_dir)[0]}


@st.cache_data(show_spinner="Reading manifests…")
def load_rows(styles_dir: str) -> pd.DataFrame:
    products = products_by_id(styles_dir)
    attributes = ontology().attributes
    splits = {int(r["product_id"]): r for r in read_csv(MANIFEST_DIR / "products.csv")}
    rows = []
    for r in read_csv(MANIFEST_DIR / "labels.csv"):
        pid = int(r["product_id"])
        p, s = products[pid], splits[pid]
        named = sorted(attributes[r["attribute"]].named_labels(matching_text(p)))
        rows.append({
            "product_id": pid, "attribute": r["attribute"], "category": s["category"], "split": s["split"],
            "family_id": int(s["family_id"]), "raw_value": r["raw_value"], "label": r["label"],
            "text_status": r["text_status"], "text_names": ", ".join(named),
            "decision": "included" if r["included"] == "1" else r["reason"], "title": p.title,
        })
    return pd.DataFrame(rows)


def queue_keys(rows: pd.DataFrame) -> list[tuple[int, str]]:
    """The frozen hand-check sample if it exists, otherwise a fresh draw."""
    if QUEUE_FILE.exists():
        return [(int(r["product_id"]), r["attribute"]) for r in read_csv(QUEUE_FILE)]
    return hand_check_queue(rows[["product_id", "attribute", "split", "text_status"]].to_dict("records"))


def load_annotations() -> dict:
    """{(product_id, attribute): {reviewer: row}}. Rows without a reviewer were saved by a person."""
    out = {}
    if ANNOTATIONS.exists():
        for r in read_csv(ANNOTATIONS):
            out.setdefault((int(r["product_id"]), r["attribute"]), {})[r.get("reviewer") or HUMAN] = r
    return out


def save_annotation(record: dict) -> None:
    annotations = load_annotations()
    annotations.setdefault((int(record["product_id"]), record["attribute"]), {})[record["reviewer"]] = record
    ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for key in sorted(annotations):
            for reviewer in sorted(annotations[key]):
                row = dict(annotations[key][reviewer], reviewer=reviewer)
                writer.writerow({k: row.get(k, "") for k in ANNOTATION_FIELDS})


def highlight(text: str, attribute, label: str) -> str:
    """Escape the text and mark every kept-label phrase it contains."""
    spans = sorted((m.start(), m.end(), lab.name) for lab in attribute.labels
                   for p in lab.patterns for m in p.finditer(text))
    out, pos = [], 0
    for start, end, name in spans:
        if start < pos:
            continue
        css = "agree" if name == label else "other"
        out += [html.escape(text[pos:start]),
                f'<mark class="{css}" title="{html.escape(name)}">{html.escape(text[start:end])}</mark>']
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def md(text) -> str:
    """Escape Markdown so product text renders literally."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", text) if isinstance(text, str) and text else ""


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "–"


st.set_page_config(page_title="Product review", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
st.title("Product review")
state = st.session_state
state.setdefault("table_version", 0)
if message := state.pop("flash", None):
    st.toast(message)


def show(key: tuple[int, str], flash: str | None = None) -> None:
    """Make `key` the current item, clear the table's own selection so it cannot override it, and rerun."""
    state.current = key
    state.table_version += 1
    if flash:
        state.flash = flash
    st.rerun()

styles_dir = next((d for d in STYLE_DIRS if d.exists()), None)
if styles_dir is None or not (MANIFEST_DIR / "labels.csv").exists():
    st.error("Product records or manifests are missing. Run the data steps in README.md first.")
    st.stop()

rows = load_rows(str(styles_dir))
rows = rows[~rows["split"].isin(TEST_SPLITS)]  # test splits stay unseen
annotations = load_annotations()
keys = queue_keys(rows)
order = {key: k for k, key in enumerate(keys, start=1)}
queue = rows[[(r.product_id, r.attribute) in order for r in rows.itertuples()]].copy()
queue["k"] = [order[(r.product_id, r.attribute)] for r in queue.itertuples()]
queue = queue.sort_values("k")


def verdict_of(row, reviewer):
    return annotations.get((row.product_id, row.attribute), {}).get(reviewer, {}).get("verdict")


def blind(row) -> bool:
    return annotations.get((row.product_id, row.attribute), {}).get(HUMAN, {}).get("blind") == "1"


with st.sidebar:
    mode = st.radio("Mode", ["Browse", "Hand-check queue"], key="mode")
    if mode == "Browse":
        attribute_key = st.radio("Attribute", list(ATTRIBUTE_NAMES), format_func=ATTRIBUTE_NAMES.get, key="attribute")
        view_id = f"browse-{attribute_key}"
        view = rows[rows["attribute"] == attribute_key]
        splits = st.multiselect("Split", sorted(view["split"].unique()), key="splits", placeholder="All splits")
        categories = st.multiselect("Category", sorted(view["category"].unique()), key="categories",
                                    placeholder="All categories")
        decisions = st.multiselect("Decision", sorted(view["decision"].unique()), key="decisions",
                                   placeholder="All decisions")
        statuses = st.multiselect("Description", list(STATUS_NAMES), format_func=STATUS_NAMES.get, key="statuses",
                                  placeholder="Any")
        query = st.text_input("Search titles", key="query")
        if splits:
            view = view[view["split"].isin(splits)]
        if categories:
            view = view[view["category"].isin(categories)]
        if decisions:
            view = view[view["decision"].isin(decisions)]
        if statuses:
            view = view[view["text_status"].isin(statuses)]
        if query:
            view = view[view["title"].str.contains(query, case=False, regex=False)]
    else:
        # Spot check: the first ten of each decisive Claude verdict, in the queue's seeded order.
        spot = {(r.product_id, r.attribute) for verdict in ("Description is wrong", "Catalog label is wrong")
                for r in [q for q in queue.itertuples() if verdict_of(q, "claude") == verdict][:SPOT_CHECK_PER_VERDICT]}
        yours = [r for r in queue.itertuples() if verdict_of(r, HUMAN)]
        both = [r for r in yours if verdict_of(r, "claude")]
        blind_both = [r for r in both if blind(r)]
        shown_both = [r for r in both if not blind(r)]
        st.progress(len(yours) / max(len(queue), 1), text=f"You reviewed {len(yours)} of {len(queue)}")
        def agreeing(items):
            return sum(verdict_of(r, HUMAN) == verdict_of(r, "claude") for r in items)

        st.caption(f"Blind agreement with Claude: {agreeing(blind_both)} of {len(blind_both)}. Earlier verdicts, "
                   f"saved with Claude's verdict shown and pre-selected: {agreeing(shown_both)} of {len(shown_both)}.")
        queue_view = st.radio("Show", QUEUE_VIEWS, key="queue_view")
        view_id = f"queue-{QUEUE_VIEWS.index(queue_view)}"
        view = queue
        if queue_view == QUEUE_VIEWS[0]:
            view = view[[not verdict_of(r, HUMAN) for r in view.itertuples()]]
        elif queue_view == QUEUE_VIEWS[1]:
            view = view[[(r.product_id, r.attribute) in spot for r in view.itertuples()]]
        elif queue_view == QUEUE_VIEWS[2]:
            view = view[[verdict_of(r, "claude") == "Can't tell" and bool(verdict_of(r, HUMAN))
                         for r in view.itertuples()]]
        elif queue_view == QUEUE_VIEWS[3]:
            view = view[[bool(verdict_of(r, HUMAN)) and bool(verdict_of(r, "claude"))
                         and verdict_of(r, HUMAN) != verdict_of(r, "claude") for r in view.itertuples()]]

cols = st.columns(4)
cols[0].metric("Rows", f"{len(view):,}")
cols[1].metric("Included", pct(int((view["decision"] == "included").sum()), len(view)))
cols[2].metric("Description names another label", pct(int((view["text_status"] == "conflicts").sum()), len(view)))
cols[3].metric("Items with a verdict", f"{len(annotations):,}")

if view.empty:
    st.info("Nothing to show here. Choose another filter.")
    st.stop()

columns = ["product_id", "attribute", "category", "split", "label", "text_names", "decision"]
if mode != "Browse":
    view = view.assign(claude=[(verdict_of(r, "claude") or "") if verdict_of(r, HUMAN) else "hidden until you save"
                               for r in view.itertuples()],
                       you=[verdict_of(r, HUMAN) or "" for r in view.itertuples()])
    columns = ["k"] + columns[:2] + ["label", "text_names", "claude", "you"]
keys_in_view = [(int(p), a) for p, a in zip(view["product_id"], view["attribute"])]
event = st.dataframe(view[columns], hide_index=True, on_select="rerun", selection_mode="single-row",
                     key=f"table-{view_id}-{state.table_version}", height=300)
if event and event.selection.rows:
    state.current = keys_in_view[event.selection.rows[0]]
if state.get("current") not in keys_in_view:
    unreviewed = [k for k in keys_in_view if HUMAN not in annotations.get(k, {})]
    state.current = unreviewed[0] if mode != "Browse" and unreviewed else keys_in_view[0]
position = keys_in_view.index(state.current)
row = view.iloc[position]

nav = st.columns([1, 1, 5])
if nav[0].button("← Previous", key="previous", disabled=position == 0):
    show(keys_in_view[position - 1])
if nav[1].button("Next →", key="next", disabled=position == len(keys_in_view) - 1):
    show(keys_in_view[position + 1])
nav[2].caption(f"Item {position + 1} of {len(keys_in_view)} in this view. Click a row in the table to jump to it.")

product = products_by_id(str(styles_dir))[int(row["product_id"])]
attribute = ontology().attributes[row["attribute"]]

st.divider()
left, right = st.columns([1, 2], gap="large")
with left:
    photo = IMAGE_DIR / f"{product.id}.jpg"
    if photo.exists():
        st.image(str(photo))
    else:
        st.info("No photo for this product.")
with right:
    st.subheader(md(product.title) or "(no title)")
    st.caption(f"ID {product.id} · {row['category']} · {row['split']} · family {row['family_id']}")
    st.markdown(f"**{ATTRIBUTE_NAMES[row['attribute']]}:** catalog says {md(row['label']) or md(row['raw_value'])}; "
                f"the description {STATUS_NAMES.get(row['text_status'], 'is not checked')}"
                + (f" ({md(row['text_names'])})" if row["text_names"] else "")
                + f". Decision: **{row['decision']}**.")
    full_tab, redacted_tab = st.tabs(["Full text", "Redacted text"])
    with full_tab:
        st.markdown(f'<div class="text-block">{highlight(product.description, attribute, row["label"])}</div>',
                    unsafe_allow_html=True)
        st.caption("Green: the description names the catalog label. Red: it names another label.")
    with redacted_tab:
        st.markdown(f"**Title:** {md(redact_title(product.title, attribute, product.brand))}")
        st.markdown(md(redact_description(product.description, attribute, product.brand)) or "(empty)")

    in_queue = (product.id, row["attribute"]) in order
    if (in_queue or row["text_status"] == "conflicts") and row["split"] in REVIEW_SPLITS:
        verdicts = annotations.get((product.id, row["attribute"]), {})
        claude, yours = verdicts.get("claude"), verdicts.get(HUMAN)
        if claude and yours:
            st.info(f"Claude's verdict: **{claude['verdict']}**. {claude.get('note', '')}")
        elif claude:
            st.caption("Claude has a verdict on this item. It appears once you save yours.")
        start = (yours or {}).get("verdict")
        with st.form(key=f"verdict-{product.id}-{row['attribute']}"):
            st.markdown("**Your verdict:** looking at the photo, which is right?")
            verdict = st.radio("Verdict", VERDICTS, key=f"verdict-choice-{product.id}-{row['attribute']}",
                               index=VERDICTS.index(start) if start in VERDICTS else None)
            note = st.text_input("Note (optional)", value=(yours or {}).get("note", ""),
                                 key=f"note-{product.id}-{row['attribute']}")
            if st.form_submit_button("Save my verdict"):
                if verdict is None:
                    st.warning("Choose a verdict first.")
                    st.stop()
                save_annotation({"product_id": product.id, "attribute": row["attribute"], "catalog_label": row["label"],
                                 "text_names": row["text_names"], "verdict": verdict, "note": note, "reviewer": HUMAN,
                                 "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 # Blind only on the first save: after it, Claude's verdict is on screen.
                                 "blind": "" if yours else "1"})
                saved = load_annotations()
                later = keys_in_view[position + 1:] + keys_in_view[:position]
                pending = [k for k in later if HUMAN not in saved.get(k, {})]
                if pending:
                    show(pending[0], "Saved. Showing the next item you haven't reviewed.")
                show(state.current, "Saved. You've reviewed every item in this view.")

family = rows[(rows["family_id"] == row["family_id"]) & (rows["attribute"] == row["attribute"])]
if len(family) > 1:
    st.markdown("**Colour variants in the same family**")
    st.dataframe(family[["product_id", "category", "split", "label", "decision"]], hide_index=True)
