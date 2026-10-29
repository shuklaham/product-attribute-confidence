# Phase 4 on the pod, unattended: one more seed of GRPO recipe v1 for M4 and M3, and of M5. Each
# model is evaluated on the five splits and the photo-swap set as soon as it is trained. M4 goes
# first, then M5, then M3: the order of the pre-registered tests, so a run cut short loses the least.
# With ABLATE it then runs the reference models it names (m0 m1 m4 m5, the seed-0 runs) on the
# photo-swap subset unswapped, once with only the photo and once with only the text.
# Expects scripts/pod/setup.sh to have run, data/ copied in (with the photo-swap subset and the M5
# data), and on the volume disk (linked as artifacts/): the M1 adapter, the v1 sampling pass
# (grpo_pool), and the adapter of every model in ABLATE. Everything it writes is under /workspace,
# the pod's volume disk, which a stop keeps.
#
# The pod stops itself:
# - when this script ends, for any reason, after up to 30 minutes for the laptop to copy the outputs;
# - when no file under /workspace/logs or /workspace/artifacts has changed for 40 minutes;
# - 12 hours after the start, whatever happens.
#
# Usage: SEED=<n> [ABLATE="m0 m1"] bash scripts/pod/phase4.sh
# PLAN_ONLY=<dir> runs the same steps anywhere, writing each command to its log under <dir>/logs
# instead of running it (tests/test_pod_scripts.py).
set -uo pipefail
seed=${SEED:?set SEED}
ablate=${ABLATE:-}

reference() {  # a reference model's adapter directory, or "" for the base model
  case $1 in
    m0) echo "" ;;
    m1) echo artifacts/m1_lr2e-4/adapter ;;
    m4 | m5) echo "artifacts/$1/adapter" ;;
    *) return 1 ;;
  esac
}
for name in $ablate; do
  reference "$name" > /dev/null || { echo "ABLATE: no reference model named $name" >&2; exit 1; }
done

if [ -n "${PLAN_ONLY:-}" ]; then
  W=$PLAN_ONLY
  py="echo python"
  mkdir -p $W/logs
  log() { echo "$*" >> $W/logs/phase4.log; }
else
  cd /root/pac || exit 1
  . scripts/pod/selfstop.sh
  export UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/root/hf HF_HUB_OFFLINE=1 PATH=/root/.local/bin:$PATH
  W=/workspace
  py="uv run --group gpu python -I"
  mkdir -p $W/logs
  log() { echo "$(date -u +%H:%M:%S) $*" >> $W/logs/phase4.log; }
  pod_api_check || { log "the pod cannot reach the Runpod API, so it could not stop itself; not starting"; exit 1; }
  rm -f $W/EXIT_STATUS $W/COPIED
  trap 'end_run "exit $?"' EXIT
  setsid nohup bash scripts/pod/guard.sh 2400 43200 $W/logs $W/artifacts > $W/logs/guard.log 2>&1 < /dev/null &
  log "started seed $seed, ablating '$ablate', at commit $(git rev-parse --short HEAD); guard pid $!"
fi

run() {  # name, command...
  local name=$1 code
  shift
  "$@" > "$W/logs/$name.log" 2>&1
  code=$?
  log "$name exit $code"
  return $code
}

# An evaluation can be rerun from the saved adapter, so a failed one does not stop what follows;
# the script still ends with a failure code.
evaluate() {  # name, adapter directory or "" for the base model; the primary split first
  local name=$1 lora="" split
  [ -n "$2" ] && lora="--lora $2"
  for split in heldout_test test cal_threshold cal_fit dev; do
    run "${name}_$split" $py scripts/predict.py "data/processed/v1/$split.jsonl" data/images \
      "artifacts/${name}_$split" $lora || failed=1
  done
}

swap() {  # name, adapter directory or "" for the base model
  local name=$1 lora=""
  [ -n "$2" ] && lora="--lora $2"
  run "${name}_swap" $py scripts/predict.py data/processed/v1/swap_heldout.jsonl data/images \
    "artifacts/${name}_swap" $lora || failed=1
}

# Training failures do not stop the run either: the other models start from M1, not from this one.
train() {  # name, training command...
  local name=$1
  shift
  if run "train_$name" "$@"; then
    evaluate "$name" "artifacts/$name/adapter"
    swap "$name" "artifacts/$name/adapter"
  else
    failed=1
  fi
}

# Photo only keeps one text condition, since without the text the two are the same input.
evidence() {  # name, adapter directory or "" for the base model
  local name=$1 lora=""
  [ -n "$2" ] && lora="--lora $2"
  run "${name}_subset_photo_only" $py scripts/predict.py data/processed/v1/subset_heldout.jsonl data/images \
    "artifacts/${name}_subset_photo_only" --mode photo_only --condition redacted $lora || failed=1
  run "${name}_subset_text_only" $py scripts/predict.py data/processed/v1/subset_heldout.jsonl data/images \
    "artifacts/${name}_subset_text_only" --mode text_only $lora || failed=1
}

failed=0
grpo="scripts/train_grpo.py data/processed/v1/train.jsonl data/images artifacts/m1_lr2e-4/adapter"
train "m4_v1_s$seed" $py $grpo "artifacts/m4_v1_s$seed" --reward calibration --seed "$seed" --pool artifacts/grpo_pool
train "m5_s$seed" $py scripts/train_sft.py data/processed/v1/calibsft.jsonl data/images "artifacts/m5_s$seed" \
  --lr 1e-4 --calibrated --init-adapter artifacts/m1_lr2e-4/adapter --seed "$seed"
train "m3_v1_s$seed" $py $grpo "artifacts/m3_v1_s$seed" --reward correctness --seed "$seed" --pool artifacts/grpo_pool
for name in $ablate; do
  evidence "$name" "$(reference "$name")"
done
log ALL_DONE
exit $failed
