#!/bin/bash
# One scheduled cycle: snapshot the newest posts, then label posts that are due.
# Runs under launchd (see scripts/schedule.sh). Safe to run by hand as well.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/opt/anaconda3/bin/python
LOCK="$ROOT/Data/.hourly.lock"

mkdir -p "$ROOT/Data"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) previous cycle still running, skipping"
    exit 0
fi
trap 'rmdir "$LOCK"' EXIT
cd "$ROOT"

echo "=== $(date -u +%FT%TZ) cycle start ==="
# 200 newest per community: two listing requests, and more than a day of posts
# even for the busiest community, so an overnight sleep loses nothing.
"$PY" scripts/collect.py --posts 200 || echo "collect failed (exit $?)"
"$PY" scripts/label.py || echo "label failed (exit $?)"
echo "=== $(date -u +%FT%TZ) cycle end ==="
