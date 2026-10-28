# Phase 1 on the pod, unattended: M0 on the calibration and test splits, LoRA SFT at three learning rates
# with dev predictions after each, the learning-rate rule, then M1 on the calibration and test splits.
# Expects the repo at /root/pac, set up by scripts/pod/setup.sh, with data/ copied in.
# This is the script that produced reports/phase1.md (run on 2026-10-26 at commit fdbc5b9).
set -uo pipefail
export UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/root/hf PATH=/root/.local/bin:$PATH
cd /root/pac
mkdir -p /root/logs
log() { echo "$(date -u +%H:%M:%S) $*"; }
predict() {  # name, split, extra args
  name=$1; split=$2; shift 2
  uv run --group gpu python -I scripts/predict.py "data/processed/v1/$split.jsonl" data/images "artifacts/$name" "$@" > "/root/logs/$name.log" 2>&1
  log "$name exit $?"
}
for split in cal_fit cal_threshold test heldout_test; do predict "m0_$split" "$split"; done
for lr in 5e-5 1e-4 2e-4; do
  uv run --group gpu python -I scripts/train_sft.py data/processed/v1/train.jsonl data/images "artifacts/m1_lr$lr" --lr "$lr" > "/root/logs/train_$lr.log" 2>&1
  log "train lr $lr exit $? $(grep -o 'trained.*' /root/logs/train_$lr.log | tail -1)"
  predict "m1_lr${lr}_dev" dev --lora "artifacts/m1_lr$lr/adapter"
done
# Learning-rate rule, fixed before training: highest pooled accuracy on redacted dev (invalid counts as wrong),
# then fewer invalid outputs, then the lower learning rate.
best=$(python3 - <<'PY'
import json, sys
scores = {}
for lr in ("5e-5", "1e-4", "2e-4"):
    rows = [json.loads(l) for l in open(f"artifacts/m1_lr{lr}_dev/predictions.jsonl")]
    rows = [r for r in rows if r["condition"] == "redacted"]
    acc = sum(r["correct"] for r in rows) / len(rows)
    invalid = sum(r["error"] is not None for r in rows) / len(rows)
    scores[lr] = (acc, -invalid, -float(lr))
    print(f"lr {lr}: redacted dev accuracy {acc:.4f}, invalid {invalid:.4f}", file=sys.stderr)
print(max(scores, key=scores.get))
PY
)
log "best lr $best"
echo "$best" > artifacts/m1_best_lr.txt
for split in cal_fit cal_threshold test heldout_test; do predict "m1_$split" "$split" --lora "artifacts/m1_lr$best/adapter"; done
log ALL_DONE
