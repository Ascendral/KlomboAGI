#!/bin/bash
# KlomboAGI Install Script — deploys as a persistent daemon on macOS
#
# Usage: ./deploy/install.sh
#
# What it does:
#   1. Copies KlomboAGI to /opt/klomboagi
#   2. Installs Python dependencies
#   3. Creates data and log directories
#   4. Installs launchd plist (auto-start on boot)
#   5. Starts the service
#
# To uninstall: ./deploy/install.sh --uninstall

set -e

INSTALL_DIR="/opt/klomboagi"
PLIST_NAME="com.klomboagi.server"
PLIST_SRC="$(dirname "$0")/${PLIST_NAME}.plist"
PLIST_DST="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Uninstall ──
if [ "$1" = "--uninstall" ]; then
    echo -e "${YELLOW}Uninstalling KlomboAGI...${NC}"
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo -e "${GREEN}Service stopped and plist removed.${NC}"
    echo -e "${YELLOW}Data preserved at ${INSTALL_DIR}/data${NC}"
    echo -e "${YELLOW}To fully remove: sudo rm -rf ${INSTALL_DIR}${NC}"
    exit 0
fi

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║     KlomboAGI — Cognitive OS Install     ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check Python ──
PYTHON=""
for p in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Error: Python 3 not found. Install with: brew install python${NC}"
    exit 1
fi

PYVERSION=$($PYTHON --version 2>&1)
echo -e "Python: ${GREEN}${PYVERSION}${NC} ($(which $PYTHON))"

# ── Create install directory ──
echo -e "\n${YELLOW}Installing to ${INSTALL_DIR}...${NC}"
sudo mkdir -p "$INSTALL_DIR"
sudo chown "$(whoami)" "$INSTALL_DIR"

# Stop running service before updating
launchctl unload "$PLIST_DST" 2>/dev/null || true
sleep 1

# Copy source (preserve data and logs from previous installs)
rsync -a --delete \
    --exclude '__pycache__' --exclude '.git' --exclude '*.pyc' \
    --exclude '.pytest_cache' --exclude '*.egg-info' --exclude 'venv' \
    --exclude 'data' --exclude 'logs' \
    "$REPO_DIR/" "$INSTALL_DIR/"

# Create directories (preserved across updates)
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"

echo -e "  Source:  ${GREEN}copied${NC}"

# ── Install dependencies ──
echo -e "\n${YELLOW}Installing dependencies...${NC}"
$PYTHON -m pip install psutil --quiet 2>/dev/null || \
    $PYTHON -m pip install --user psutil --quiet 2>/dev/null || \
    $PYTHON -m pip install --break-system-packages psutil --quiet 2>/dev/null || true
echo -e "  psutil:  ${GREEN}installed${NC}"

# ── Test import ──
echo -e "\n${YELLOW}Testing brain import...${NC}"
cd "$INSTALL_DIR"
if $PYTHON -c "from klomboagi.core.genesis import Genesis; print('Brain OK')" 2>/dev/null; then
    echo -e "  Genesis: ${GREEN}OK${NC}"
else
    echo -e "  Genesis: ${RED}FAILED${NC}"
    echo "  Check the error above. The service may not start."
fi

# ── Test hardware sense ──
if $PYTHON -c "
from klomboagi.senses.hardware import HardwareSense
hw = HardwareSense().scan()
print(f'Hardware: {hw.cpu.model}, {hw.ram.total_gb:.0f}GB RAM')
" 2>/dev/null; then
    echo -e "  Hardware: ${GREEN}OK${NC}"
fi

# ── Update plist with correct Python path ──
PYTHON_PATH=$(which $PYTHON)
echo -e "\n${YELLOW}Installing launchd service...${NC}"

# Create plist with correct Python path
cat > "$PLIST_DST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>-m</string>
        <string>klomboagi</string>
        <string>serve</string>
        <string>--port</string>
        <string>3141</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>${INSTALL_DIR}/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${INSTALL_DIR}/logs/stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>${INSTALL_DIR}</string>
        <key>KLOMBOAGI_HOME</key>
        <string>${INSTALL_DIR}/data</string>
    </dict>
    <key>ProcessType</key>
    <string>Background</string>
    <key>LowPriorityBackgroundIO</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
PLISTEOF

echo -e "  Plist:   ${GREEN}${PLIST_DST}${NC}"

# ── Start the service ──
echo -e "\n${YELLOW}Starting KlomboAGI...${NC}"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

# Wait for server to start (Genesis brain takes a few seconds to load)
echo -n "  Waiting"
for i in $(seq 1 15); do
    if curl -s http://localhost:3141/health 2>/dev/null | grep -q "alive"; then
        break
    fi
    echo -n "."
    sleep 1
done
echo

# Check if it's running
if curl -s http://localhost:3141/health | grep -q "alive" 2>/dev/null; then
    echo -e "  Status:  ${GREEN}RUNNING${NC}"
else
    echo -e "  Status:  ${YELLOW}Starting... (check logs)${NC}"
fi

# ── Install menu bar app ──
MENUBAR_PLIST="$HOME/Library/LaunchAgents/com.klomboagi.menubar.plist"
echo -e "\n${YELLOW}Installing menu bar app...${NC}"

launchctl unload "$MENUBAR_PLIST" 2>/dev/null || true

cat > "$MENUBAR_PLIST" << MBEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.klomboagi.menubar</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>-m</string>
        <string>klomboagi</string>
        <string>menubar</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${INSTALL_DIR}/logs/menubar_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${INSTALL_DIR}/logs/menubar_stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>${INSTALL_DIR}</string>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
MBEOF

launchctl load "$MENUBAR_PLIST"
echo -e "  Menu bar: ${GREEN}installed${NC}"

# ── Open dashboard ──
sleep 2
open "http://localhost:3141" 2>/dev/null || true

echo -e "\n${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  KlomboAGI is alive                      ║${NC}"
echo -e "${GREEN}║  🧠 Menu bar icon active                  ║${NC}"
echo -e "${GREEN}║  🌐 Dashboard: http://localhost:3141       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo
echo "  Chat:        python3 -m klomboagi chat"
echo "  Dashboard:   http://localhost:3141"
echo "  Logs:        tail -f ${INSTALL_DIR}/logs/stdout.log"
echo "  Uninstall:   ${REPO_DIR}/deploy/install.sh --uninstall"
