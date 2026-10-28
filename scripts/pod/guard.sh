# Ends the run and stops the pod if the run stalls or overruns. phase2.sh starts it in the background.
# Usage: guard.sh <stall_seconds> <limit_seconds> <dir>...
# A stall is no file under the given directories changing for stall_seconds.
. "$(dirname "$0")/selfstop.sh"
stall=$1 limit=$2
shift 2
start=$(date +%s)
while sleep 60; do
  now=$(date +%s)
  if [ $((now - start)) -gt "$limit" ]; then
    end_run "guard: time limit of $limit s reached"
    exit
  fi
  newest=$(find "$@" -type f -printf '%T@\n' 2> /dev/null | sort -n | tail -n 1)
  if [ -n "$newest" ] && [ $((now - ${newest%.*})) -gt "$stall" ]; then
    end_run "guard: no file changed for $stall s"
    exit
  fi
done
