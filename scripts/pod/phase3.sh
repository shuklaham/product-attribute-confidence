# Phase 3 on the pod, unattended: GRPO recipe v2 for one seed, M4 then M3, each evaluated on the
# five splits and the photo-swap set as soon as it is trained. With EXTRA=1 it then trains and
# evaluates M5, and runs the photo-swap set for M0 and M1.
# Expects scripts/pod/setup.sh to have run, data/ copied in (with the photo-swap and M5 files) and
# the M1 adapter on the volume disk (linked as artifacts/). Everything it writes is under /workspace,
# the pod's volume disk, which a stop keeps.
#
# The pod stops itself:
# - when this script ends, for any reason, after up to 30 minutes for the laptop to copy the outputs;
# - when no file under /workspace/logs or /workspace/artifacts has changed for 40 minutes;
# - 15 hours after the start, whatever happens.
#
# Usage: SEED=<n> [RESCORE=50] [EXTRA=1] bash scripts/pod/phase3.sh
# RESCORE is the steps between re-scorings: 50, or 100 if the pre-flight check measured a full
# scoring pass above 6 minutes (docs/plan.md, decision log 2026-10-28).
# PLAN_ONLY=<dir> runs the same steps anywhere, writing each command to its log under <dir>/logs
# instead of running it (tests/test_pod_scripts.py).
set -uo pipefail
seed=${SEED:?set SEED}
rescore=${RESCORE:-50}
if [ -n "${PLAN_ONLY:-}" ]; then
  W=$PLAN_ONLY
  py="echo python"
  mkdir -p $W/logs
  log() { echo "$*" >> $W/logs/phase3.log; }
else
  cd /root/pac || exit 1
  . scripts/pod/selfstop.sh
  export UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/root/hf HF_HUB_OFFLINE=1 PATH=/root/.local/bin:$PATH
  W=/workspace
  py="uv run --group gpu python -I"
  mkdir -p $W/logs
  log() { echo "$(date -u +%H:%M:%S) $*" >> $W/logs/phase3.log; }
  pod_api_check || { log "the pod cannot reach the Runpod API, so it could not stop itself; not starting"; exit 1; }
  rm -f $W/EXIT_STATUS $W/COPIED
  trap 'end_run "exit $?"' EXIT
  setsid nohup bash scripts/pod/guard.sh 2400 54000 $W/logs $W/artifacts > $W/logs/guard.log 2>&1 < /dev/null &
  log "started seed $seed at commit $(git rev-parse --short HEAD); guard pid $!"
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

failed=0
for arm in "m4 calibration" "m3 correctness"; do
  set -- $arm
  name="${1}_v2_s$seed" kind=$2
  run "train_$name" $py scripts/train_grpo.py data/processed/v1/train.jsonl data/images \
    artifacts/m1_lr2e-4/adapter "artifacts/$name" --reward "$kind" --seed "$seed" --rescore-every "$rescore" || exit 1
  evaluate "$name" "artifacts/$name/adapter"
  swap "$name" "artifacts/$name/adapter"
done
if [ -n "${EXTRA:-}" ]; then
  if run train_m5 $py scripts/train_sft.py data/processed/v1/calibsft.jsonl data/images artifacts/m5 \
    --lr 1e-4 --calibrated --init-adapter artifacts/m1_lr2e-4/adapter; then
    evaluate m5 artifacts/m5/adapter
    swap m5 artifacts/m5/adapter
  else
    failed=1
  fi
  swap m0 ""
  swap m1 artifacts/m1_lr2e-4/adapter
fi
log ALL_DONE
exit $failed
