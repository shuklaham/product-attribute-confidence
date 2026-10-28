#!/usr/bin/env bash
# Copies what a pod run needs from the laptop:
# - the repo's tracked files and its git history, nothing else, so untracked files such as .env and
#   private/ can never reach a pod;
# - the built examples and photos;
# - the M1 adapter, onto the pod's volume disk, which is linked as the repo's artifacts/.
# Usage: scripts/pod/push.sh <host> <port> <known_hosts>
set -euo pipefail
host=$1 port=$2 known=$3
ssh_cmd="ssh -i $HOME/.ssh/id_ed25519 -p $port -o UserKnownHostsFile=$known -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -o BatchMode=yes"
cd "$(dirname "$0")/../.."
$ssh_cmd root@"$host" 'mkdir -p /root/pac /workspace/artifacts/m1_lr2e-4'
git ls-files -z | rsync -az --from0 --files-from=- -e "$ssh_cmd" ./ root@"$host":/root/pac/
rsync -az -e "$ssh_cmd" .git/ root@"$host":/root/pac/.git/
rsync -azR -e "$ssh_cmd" data/processed/v1 data/images root@"$host":/root/pac/
# The volume disk is a network filesystem that refuses chown, so ownership is not copied there.
rsync -rltz -e "$ssh_cmd" artifacts/m1_lr2e-4/adapter root@"$host":/workspace/artifacts/m1_lr2e-4/
# rsync keeps the laptop's file owner, which git on the pod refuses to work with.
# shellcheck disable=SC2016  # the command runs on the pod
$ssh_cmd root@"$host" 'chown -R root:root /root/pac && ln -sfn /workspace/artifacts /root/pac/artifacts &&
  cd /root/pac && git status --short | head -5 && echo "pushed $(git rev-parse --short HEAD)"'
