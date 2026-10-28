# Sourced by scripts on the pod so they can stop their own pod.
# Runpod writes the pod's id and an API key for it to /etc/rp_environment, which non-interactive SSH
# sessions do not read, so it is read here. Checked on a live pod (2026-10-28): that key works with
# Runpod's GraphQL API for this pod, while runpodctl 1.14 and the REST API refuse it, so the calls
# go to GraphQL directly.
[ -f /etc/rp_environment ] && . /etc/rp_environment

graphql() {
  curl -s --max-time 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\":\"$1\"}" https://api.runpod.io/graphql
}

# Stops this pod, retrying for a few minutes. A stop ends GPU billing and keeps the volume disk (/workspace).
stop_pod() {
  for _ in 1 2 3 4 5 6; do
    graphql "mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }" |
      grep -q '"desiredStatus"' && return 0
    sleep 30
  done
  return 1
}

# Succeeds only if the pod can read itself through the API; checked before any long work.
pod_api_check() {
  [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || return 1
  graphql "query { pod(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }" | grep -q '"desiredStatus"'
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
