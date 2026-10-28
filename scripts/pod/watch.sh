#!/usr/bin/env bash
# Watches an unattended pod run from the laptop.
# - Every 5 minutes (POLL seconds, if set) it copies /workspace/artifacts back, and the pod's logs into _pod_logs.
# - When the run has ended (/workspace/EXIT_STATUS exists), it makes a last copy, checks it against
#   checksums made on the pod, and marks /workspace/COPIED, so the pod stops at once.
# - Every step is a short SSH call that may fail and is retried, so a network drop or a sleeping
#   laptop only delays the copy; the pod stops itself regardless (scripts/pod/phase2.sh).
# Exits 0 once the outputs are copied and verified; 2 if the pod, once reachable, stays unreachable
# for 30 minutes (look at it with the Runpod tools); 1 at the deadline.
# Usage: scripts/pod/watch.sh <host> <port> <known_hosts> <local_dir> [deadline_hours]
set -uo pipefail
host=$1 port=$2 known=$3 dest=$4 hours=${5:-14}
ssh_cmd="ssh -i $HOME/.ssh/id_ed25519 -p $port -o UserKnownHostsFile=$known -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"
on_pod() { $ssh_cmd root@"$host" "$1"; }
stamp() { date -u +%H:%M; }
mkdir -p "$dest"
deadline=$(($(date +%s) + hours * 3600))
last_seen=$(date +%s)
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! status=$(on_pod 'cat /workspace/EXIT_STATUS 2> /dev/null; true'); then
    if [ $(($(date +%s) - last_seen)) -gt 1800 ]; then
      echo "$(stamp) pod unreachable for 30 minutes"
      exit 2
    fi
    sleep 60
    continue
  fi
  last_seen=$(date +%s)
  if ! rsync -a --timeout=120 -e "$ssh_cmd" root@"$host":/workspace/artifacts/ "$dest/"; then
    echo "$(stamp) copy failed; retrying"
    sleep 60
    continue
  fi
  # The pod's logs too (timings, exit codes, tracebacks); they are not checksummed, as they keep growing.
  rsync -a --timeout=120 -e "$ssh_cmd" root@"$host":/workspace/logs/ "$dest/_pod_logs/" || true
  if [ -n "$status" ]; then
    if on_pod 'cd /workspace/artifacts && find . -type f -print0 | sort -z | xargs -0 sha256sum' > "$dest/.pod.sha256" &&
      (cd "$dest" && shasum -a 256 -c --quiet .pod.sha256) && on_pod 'touch /workspace/COPIED'; then
      echo "$(stamp) run ended ($(echo "$status" | tr '\n' ';')); outputs copied and verified"
      exit 0
    fi
    echo "$(stamp) final check failed; retrying"
    sleep 60
    continue
  fi
  echo "$(stamp) $(on_pod 'tail -n 1 /workspace/logs/phase2.log' 2> /dev/null)"
  sleep "${POLL:-300}"
done
echo "$(stamp) deadline passed"
exit 1
