# Phase 2 on the pod, unattended: GRPO from M1 for M4, then M3, each evaluated as soon as it is
# trained, so the primary comparison is ready first if anything cuts the run short.
# Expects scripts/pod/setup.sh to have run, data/ copied in, and on the volume disk (linked as
# artifacts/) the M1 adapter and the GRPO pool from scripts/sample_pool.py. Everything it writes is
# under /workspace, the pod's volume disk, which a stop keeps.
#
# The pod stops itself:
# - when this script ends, for any reason, after up to 30 minutes for the laptop to copy the outputs;
# - when no file under /workspace/logs or /workspace/artifacts has changed for 40 minutes;
# - 12 hours after the start, whatever happens.
set -uo pipefail
cd /root/pac || exit 1
. scripts/pod/selfstop.sh
export UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/root/hf HF_HUB_OFFLINE=1 PATH=/root/.local/bin:$PATH
W=/workspace
mkdir -p $W/logs
log() { echo "$(date -u +%H:%M:%S) $*" >> $W/logs/phase2.log; }
pod_api_check || { log "the pod cannot reach the Runpod API, so it could not stop itself; not starting"; exit 1; }
rm -f $W/EXIT_STATUS $W/COPIED
trap 'end_run "exit $?"' EXIT
setsid nohup bash scripts/pod/guard.sh 2400 43200 $W/logs $W/artifacts > $W/logs/guard.log 2>&1 < /dev/null &
log "started at commit $(git rev-parse --short HEAD); guard pid $!"

run() {  # name, command...
  name=$1
  shift
  "$@" > "$W/logs/$name.log" 2>&1
  code=$?
  log "$name exit $code"
  return $code
}

for arm in "m4 calibration" "m3 correctness"; do
  set -- $arm
  name=$1 kind=$2
  run "train_$name" uv run --group gpu python -I scripts/train_grpo.py data/processed/v1/train.jsonl data/images \
    artifacts/grpo_pool artifacts/m1_lr2e-4/adapter "artifacts/$name" --reward "$kind" || exit 1
  for split in dev cal_fit cal_threshold test heldout_test; do
    run "${name}_$split" uv run --group gpu python -I scripts/predict.py "data/processed/v1/$split.jsonl" data/images \
      "artifacts/${name}_$split" --lora "artifacts/$name/adapter" || exit 1
  done
done
log ALL_DONE
