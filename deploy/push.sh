#!/bin/bash
# Push KlomboAGI to remote MacBook Air and restart service.
# Usage: ./deploy/push.sh [host] [user] [password]

HOST="${1:-192.168.68.53}"
USER="${2:-zanderpink}"
PASS="${3:-liam2016}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Deploying KlomboAGI to ${USER}@${HOST}..."

# Step 1: rsync source to home dir (no sudo needed)
expect -c "
set timeout 120
spawn rsync -az --delete \
    --exclude __pycache__ --exclude .git --exclude .pytest_cache \
    --exclude \"*.pyc\" --exclude \"*.egg-info\" --exclude data --exclude logs \
    -e \"ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no\" \
    ${REPO_DIR}/klomboagi/ ${USER}@${HOST}:~/klomboagi_staging/
expect {
    \"*assword*\" { send \"${PASS}\r\"; exp_continue }
    eof { }
}
"

# Step 2: also push deploy dir
expect -c "
set timeout 30
spawn rsync -az -e \"ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no\" \
    ${REPO_DIR}/deploy/ ${USER}@${HOST}:~/klomboagi_deploy/
expect {
    \"*assword*\" { send \"${PASS}\r\"; exp_continue }
    eof { }
}
"

# Step 3: SSH in and move files + restart
expect -c "
set timeout 60
spawn ssh -t -o PreferredAuthentications=password -o PubkeyAuthentication=no ${USER}@${HOST}
expect \"*assword*\" { send \"${PASS}\r\" }
expect \"*\\\$*\"
send \"echo ${PASS} | sudo -S rsync -a ~/klomboagi_staging/ /opt/klomboagi/klomboagi/ && echo COPIED\r\"
expect \"COPIED\"
send \"echo ${PASS} | sudo -S chown -R ${USER} /opt/klomboagi && echo OWNED\r\"
expect \"OWNED\"
send \"launchctl unload ~/Library/LaunchAgents/com.klomboagi.server.plist 2>/dev/null; sleep 2; launchctl load ~/Library/LaunchAgents/com.klomboagi.server.plist && echo RESTARTED\r\"
expect \"RESTARTED\"
send \"sleep 3 && curl -s http://localhost:3141/health && echo && exit\r\"
expect eof
"

echo ""
echo "Deploy complete."
