"""Review app for the Myntra product records.

Browse products by split, category and label, see where the description names a
neckline or sleeve-length label, and compare a product with its colour variants.

Run from the repo root:
    uv run streamlit run app/review.py
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit  # noqa: E402

STYLES_DIR = ROOT / "data" / "raw" / "myntra_mirror" / "styles"
IMAGE_DIRS = [
    ROOT / "data" / "raw" / "kaggle" / "fashion-dataset" / "images",
    ROOT / "data" / "raw" / "kaggle" / "images",
    ROOT / "data" / "images",
]
ATTRIBUTE_NAMES = {"neckline": "Neckline", "sleeve_length": "Sleeve length"}
STATUS_NAMES = {"agrees": "Text agrees", "conflicts": "Text conflicts", "silent": "Text silent"}
STATUS_HELP = {
    "agrees": "The description names the catalog label.",
    "conflicts": "The description names a different label. These are excluded from training and the clean test set.",
    "silent": "The description names no label for this attribute.",
}
STYLE = """
<style>
mark.agree { background: rgba(46, 160, 67, 0.28); color: inherit; padding: 0 2px; border-radius: 3px; }
mark.other { background: rgba(218, 54, 51, 0.28); color: inherit; padding: 0 2px; border-radius: 3px; }
.desc { line-height: 1.65; max-width: 78ch; }
</style>
"""


@st.cache_data(show_spinner="Loading product records…")
def load_products(styles_dir: str) -> pd.DataFrame:
    products, _ = audit.load(styles_dir)
    family_of = audit.families(products)
    rows = []
    for p in products:
        if p["category"] not in audit.SEEN + audit.HELD_OUT:
            continue
        text = " ".join((p["title"], p["variant"], p["description"]))
        row = {
            "id": p["id"],
            "category": p["category"],
            "split": "seen" if p["category"] in audit.SEEN else "held-out",
            "title": p["title"],
            "variant": p["variant"],
            "description": p["description"],
            "views": ", ".join(p["views"]),
            "family": family_of[p["id"]],
        }
        for attribute in audit.ATTRIBUTES:
            label = p["labels"][attribute]
            named = sorted(lab for lab, pat in audit.LABEL_PATTERNS[attribute].items() if pat.search(text))
            row[attribute] = label
            row[f"{attribute}_named"] = ", ".join(named)
            if label is None:
                row[f"{attribute}_status"] = None
            elif label in named:
                row[f"{attribute}_status"] = "agrees"
            elif named:
                row[f"{attribute}_status"] = "conflicts"
            else:
                row[f"{attribute}_status"] = "silent"
        rows.append(row)
    df = pd.DataFrame(rows)
    df["family_size"] = df.groupby("family")["id"].transform("size")
    return df


def highlight(text: str, attribute: str, label: str | None) -> str:
    """Escape the description and mark every label phrase it contains."""
    spans = sorted(
        (m.start(), m.end(), lab)
        for lab, pat in audit.LABEL_PATTERNS[attribute].items()
        for m in pat.finditer(text)
    )
    out, pos = [], 0
    for start, end, lab in spans:
        if start < pos:
            continue
        css = "agree" if lab == label else "other"
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark class="{css}" title="{html.escape(lab)}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def present(value) -> bool:
    return isinstance(value, str) and value != ""


def md(text) -> str:
    """Escape Markdown so product text renders literally."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", text) if present(text) else ""


def find_image(product_id: int) -> Path | None:
    for folder in IMAGE_DIRS:
        path = folder / f"{product_id}.jpg"
        if path.exists():
            return path
    return None


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "–"


st.set_page_config(page_title="Product review", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
st.title("Product review")

if not STYLES_DIR.exists():
    st.error(f"No product records at {STYLES_DIR.relative_to(ROOT)}. Run the download command in README.md first.")
    st.stop()

df = load_products(str(STYLES_DIR))

with st.sidebar:
    attribute = st.radio("Attribute", list(ATTRIBUTE_NAMES), format_func=ATTRIBUTE_NAMES.get, key="attribute")
    splits = st.multiselect("Split", ["seen", "held-out"], default=["seen", "held-out"], key="splits")
    in_split = df[df["split"].isin(splits)]
    labeled = in_split[in_split[attribute].notna()]
    categories = st.multiselect("Category", sorted(labeled["category"].unique()), key="categories",
                                placeholder="All categories")
    labels = st.multiselect("Catalog label", labeled[attribute].value_counts().index.tolist(), key="labels",
                            placeholder="All labels")
    statuses = st.multiselect("Description vs. label", list(STATUS_NAMES), format_func=STATUS_NAMES.get,
                              key="statuses", placeholder="Any",
                              help="\n\n".join(f"**{STATUS_NAMES[k]}**: {v}" for k, v in STATUS_HELP.items()))
    query = st.text_input("Search title or description", key="query")

view = labeled
if categories:
    view = view[view["category"].isin(categories)]
if labels:
    view = view[view[attribute].isin(labels)]
if statuses:
    view = view[view[f"{attribute}_status"].isin(statuses)]
if query:
    q = query.lower()
    view = view[view["title"].str.lower().str.contains(q, regex=False)
                | view["description"].str.lower().str.contains(q, regex=False)]

counts = view[f"{attribute}_status"].value_counts()
cols = st.columns(4)
cols[0].metric("Labeled products", f"{len(view):,}")
for col, status in zip(cols[1:], STATUS_NAMES):
    col.metric(STATUS_NAMES[status], pct(int(counts.get(status, 0)), len(view)), help=STATUS_HELP[status])

with st.expander("Label-noise estimate by category"):
    noise = (labeled.assign(conflict=labeled[f"{attribute}_status"].eq("conflicts"))
             .groupby(["split", "category"])
             .agg(labeled=("id", "size"), conflicts=("conflict", "sum"))
             .reset_index())
    noise["conflict rate"] = (noise["conflicts"] / noise["labeled"]).map("{:.0%}".format)
    st.dataframe(noise.sort_values("conflicts", ascending=False), hide_index=True)
    st.caption("Share of labeled products whose description names a different label. Some are phrase-matching "
               "misses, so treat the rate as an upper bound until the hand check.")

if view.empty:
    st.info("No products match these filters.")
    st.stop()

table = view[["id", "category", "title", attribute, f"{attribute}_named", f"{attribute}_status", "family_size"]]
table = table.rename(columns={attribute: "catalog label", f"{attribute}_named": "text names",
                              f"{attribute}_status": "status", "family_size": "family size"})
event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="table", height=320)
selected = event.selection.rows if event and event.selection.rows else [0]
product = view.iloc[selected[0]]

st.divider()
left, right = st.columns([1, 2], gap="large")
with left:
    image = find_image(int(product["id"]))
    if image:
        st.image(str(image))
    else:
        st.info("Photo not downloaded yet. It arrives with the Kaggle download.")
    st.caption(f"Views listed in the record: {product['views'] or 'none'}")
with right:
    st.subheader(md(product["title"]) or "(no title)")
    st.caption(f"ID {product['id']} · {product['category']} · {product['split']} · "
               f"family of {product['family_size']}")
    for attr, name in ATTRIBUTE_NAMES.items():
        status = product[f"{attr}_status"]
        st.markdown(f"**{name}:** {md(product[attr]) if present(product[attr]) else 'unlabeled'}"
                    + (f" · {STATUS_NAMES[status].lower()}" if present(status) else ""))
    st.markdown(f"**Variant name:** {md(product['variant']) or 'none'}")
    st.markdown(f'<div class="desc">{highlight(product["description"], attribute, product[attribute])}</div>',
                unsafe_allow_html=True)
    st.caption("Green: the description names the catalog label. Red: it names another label.")

family = df[df["family"] == product["family"]]
if len(family) > 1:
    st.markdown("**Colour variants in the same family**")
    st.dataframe(family[["id", "category", "title", "neckline", "sleeve_length"]], hide_index=True)
