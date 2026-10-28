# Sourced by scripts on the pod so they can stop their own pod.
# Runpod writes the pod's id and an API key for it to /etc/rp_environment, which non-interactive SSH
# sessions do not read, so it is read here.
[ -f /etc/rp_environment ] && . /etc/rp_environment

# Stops this pod. A stop ends GPU billing and keeps the volume disk (/workspace).
stop_pod() {
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
}

# Succeeds only if the pod can read itself through the Runpod API; checked before any long work.
pod_api_check() {
  [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl > /dev/null || return 1
  runpodctl pod get "$RUNPOD_POD_ID" > /dev/null 2>&1 || runpodctl get pod "$RUNPOD_POD_ID" > /dev/null 2>&1
}

# Records why the run ended, gives the laptop up to COPY_WAIT seconds (30 minutes unless set) to copy
# the outputs (scripts/pod/watch.sh marks /workspace/COPIED when it has), then stops the pod.
end_run() {
  echo "$1" >> /workspace/EXIT_STATUS
  for _ in $(seq $((${COPY_WAIT:-1800} / 10))); do
    [ -e /workspace/COPIED ] && break
    sleep 10
  done
  stop_pod
}
