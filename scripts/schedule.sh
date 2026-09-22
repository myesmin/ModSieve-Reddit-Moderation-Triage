#!/bin/bash
# Install, remove, or inspect the hourly collection job (macOS launchd).
#
#   bash scripts/schedule.sh install
#   bash scripts/schedule.sh status
#   bash scripts/schedule.sh uninstall
#
# launchd runs the job every hour while the Mac is awake. A sleeping Mac skips
# cycles; posts first seen late are excluded from the removal rate by design.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.myesmin.reddit-triage.hourly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
  install)
    mkdir -p "$ROOT/logs" "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/hourly.sh</string>
  </array>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/hourly.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/hourly.log</string>
</dict>
</plist>
PLIST
    launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "installed: runs now, then every hour. Log: $ROOT/logs/hourly.log"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed"
    ;;
  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      launchctl print "$DOMAIN/$LABEL" | grep -E "^\s*(state|runs|last exit code)" || true
    else
      echo "not installed"
    fi
    echo "--- last log lines ---"
    tail -n 8 "$ROOT/logs/hourly.log" 2>/dev/null || echo "(no log yet)"
    ;;
  *)
    echo "usage: bash scripts/schedule.sh install|status|uninstall" >&2
    exit 1
    ;;
esac
