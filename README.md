# Product Attribute Confidence

Does RL with a calibration reward make a vision-language model's stated confidence more trustworthy on product categories it never trained on, compared with cheap post-hoc calibration?

## Data

Product records come from the Myntra Fashion Product Images dataset (Kaggle, `paramaggarwal/fashion-product-images-dataset`). Raw data lives in `data/` and is never committed.

The exploratory audit currently reads the style JSONs from a GitHub mirror of that dataset:

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/jlarosa02/AI25_Fashion.git data/raw/myntra_mirror
git -C data/raw/myntra_mirror sparse-checkout set styles
```

Frozen manifests for the real experiment should come from the official Kaggle download. For that, copy `.env.example` to `.env` (git ignores it) and paste a token from https://www.kaggle.com/settings/api.

## Data audit

```
uv run python -I src/audit.py data/raw/myntra_mirror/styles reports/data_audit.md
```

Writes label coverage by category, label support across the seen / held-out split, a category-majority baseline, how often the text names the label, and colour-variant family statistics.

## Review app

```
uv run streamlit run app/review.py
```

Browse products by split, category, label and whether the description agrees with the catalog label. Label phrases are highlighted in the description, colour variants are listed together, and an expander shows the label-noise estimate per category. Photos appear once the Kaggle images are under `data/raw/kaggle/`.
