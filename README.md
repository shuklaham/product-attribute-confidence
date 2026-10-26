# Product Attribute Confidence

Does RL with a calibration reward make a vision-language model's stated confidence more trustworthy on product categories it never trained on, compared with cheap post-hoc calibration? The full design, decisions and milestones are in [docs/plan.md](docs/plan.md). How each part works, and why it is built that way, is in [docs/concepts.md](docs/concepts.md). The dated run log is [docs/log.md](docs/log.md).

## Setup

```
uv sync
cp .env.example .env    # then paste a Kaggle API token; .env is gitignored
```

## Data pipeline

Raw data, photos and generated examples live under `data/` and are never committed. The manifests in `manifests/v0/` hold product IDs, splits and label decisions only, with hashes of the source data and the label set.

```
# 1. Official dataset (about 25 GB) and its product records
uv run --env-file .env --with kaggle kaggle datasets download paramaggarwal/fashion-product-images-dataset -p data/raw/kaggle
unzip -q -j data/raw/kaggle/fashion-product-images-dataset.zip "fashion-dataset/styles/*" -d data/raw/kaggle/styles

# 2. Splits and label decisions, then photos (checked and resized to 512 px), then the decisions again with the photo rule
uv run python -I scripts/build_manifests.py data/raw/kaggle/styles manifests/v0
uv run python -I scripts/prepare_images.py data/raw/kaggle/fashion-product-images-dataset.zip manifests/v0 data/images reports/image_check.md
uv run python -I scripts/build_manifests.py data/raw/kaggle/styles manifests/v0

# 3. Model-facing examples in both text conditions
uv run python -I scripts/build_examples.py data/raw/kaggle/styles manifests/v0 data/processed/v0

# 4. Reports: data audit, and the CPU text-only leak check on train and dev
uv run python -I scripts/audit.py data/raw/kaggle/styles reports/data_audit.md manifests/v0/photos.csv
uv run python -I scripts/leak_check.py data/processed/v0
```

The label set, aliases and redaction phrases are in [configs/ontology.yaml](configs/ontology.yaml). Changing it changes its hash, and `build_examples.py` refuses to run until the manifests are rebuilt.

## Model runs

Inference runs on a Linux GPU pod; the tables can be made anywhere.

```
# On the pod: vLLM 0.32 and torch 2.13 (CUDA 13 wheels; NVIDIA driver 580 or newer)
uv sync --group gpu

# A smoke test, then the base model on dev in both text conditions, then the text-only redaction check
uv run --group gpu python -I scripts/predict.py data/processed/v0/dev.jsonl data/images artifacts/m0_dev_smoke --limit 16
uv run --group gpu python -I scripts/predict.py data/processed/v0/dev.jsonl data/images artifacts/m0_dev
uv run --group gpu python -I scripts/predict.py data/processed/v0/dev.jsonl data/images artifacts/m0_dev_text_only --mode text_only --condition redacted

# Metric tables
uv run python -I scripts/evaluate.py artifacts/m0_dev/predictions.jsonl artifacts/m0_dev_text_only/predictions.jsonl \
    --train data/processed/v0/train.jsonl --out reports/m0_dev.md
```

Each run writes `predictions.jsonl` and a run record (`run.json`: git commit, model revision, package versions, GPU, time) under `artifacts/`, which is gitignored. Only the aggregated tables in `reports/` are committed.

## Review app

```
uv run streamlit run app/review.py
```

Browse products by split, category, label and decision, compare full and redacted text next to the photo, and work through the frozen hand-check queue of sampled contradictions between descriptions and catalog labels. Verdicts are saved to `annotations/conflict_check.csv` with the reviewer recorded, so Claude's verdicts and a person's can be compared.

## Tests

```
uv run pytest
```

The tests use small synthetic products, so they run without the dataset. They cover label mapping, split integrity, redaction rules, prompt leakage, strict output parsing, reward values, the metrics and calibration fits, and reading answers and letter probabilities from model output. CI runs them on every push.

## Layout

| Path | Contents |
|---|---|
| `src/pac/` | Library: ontology, records, splits, redaction, examples, prompt, output parser, rewards, metrics, calibration, inference, evaluation |
| `scripts/` | Pipeline steps, model runs and reports |
| `configs/` | Label set and phrase lists |
| `manifests/` | Frozen splits and label decisions (IDs and hashes only) |
| `reports/` | Data audit and image check |
| `annotations/` | Hand-check verdicts |
| `app/` | Review app |
| `docs/` | Plan, concepts and run log |
