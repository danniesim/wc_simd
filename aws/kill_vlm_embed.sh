#!/bin/bash

# Script: kill_vlm_embed.sh
# Purpose: Terminate screen sessions and processes started by vlm_embed.sh
# Default target session name: vlm_all
# Usage:
#   ./kill_vlm_embed.sh                # kills session named 'vlm_all'
#   ./kill_vlm_embed.sh other_name     # kills session(s) matching 'other_name'
#
# What it does:
# 1. Finds screen sessions whose name matches the provided pattern (default: vlm_all) and sends them a quit command.
# 2. As a safety net, kills any remaining processes running `python -m wc_simd.vlm_embed`.
# 3. Kills any lingering `watch -n 1 nvidia-smi` process spawned by the original script.
# 4. Reports what it did.
#
# Exit codes:
#  0 = success (even if nothing was running)
#  1 = unexpected error

set -euo pipefail

SESSION_PATTERN="${1:-vlm_all}"

echo "[kill_vlm_embed] Target session pattern: ${SESSION_PATTERN}"

if ! command -v screen >/dev/null 2>&1; then
  echo "[kill_vlm_embed] ERROR: 'screen' command not found in PATH." >&2
  exit 1
fi

SCREEN_LS_OUTPUT=$(screen -ls 2>&1 || true)

if echo "$SCREEN_LS_OUTPUT" | grep -q "No Sockets found"; then
  echo "[kill_vlm_embed] No screen sessions exist. Skipping screen termination step."
else
  # Extract matching session identifiers like 12345.vlm_all
  mapfile -t MATCHING_SESSIONS < <(echo "$SCREEN_LS_OUTPUT" | grep -Eo "[0-9]+\\.${SESSION_PATTERN}" || true)

  if [ ${#MATCHING_SESSIONS[@]} -eq 0 ]; then
    echo "[kill_vlm_embed] No screen sessions matched pattern '${SESSION_PATTERN}'."
  else
    echo "[kill_vlm_embed] Found ${#MATCHING_SESSIONS[@]} screen session(s) to terminate: ${MATCHING_SESSIONS[*]}"
    for S in "${MATCHING_SESSIONS[@]}"; do
      echo "[kill_vlm_embed] Sending quit to screen session: $S"
      # Use full identifier (pid.name) for unambiguous targeting
      screen -S "$S" -X quit || echo "[kill_vlm_embed] Warning: could not quit $S (may have already exited)"
    done
  fi
fi

echo "[kill_vlm_embed] Killing residual vlm_embed python processes (if any)..."
pkill -f 'python -m wc_simd.vlm_embed' 2>/dev/null || true

echo "[kill_vlm_embed] Killing residual GPU monitor (watch nvidia-smi) processes (if any)..."
pkill -f 'watch -n 1 nvidia-smi' 2>/dev/null || true

# Give processes a brief moment to terminate
sleep 0.5

# Report remaining processes (if any) for visibility
REMAINING=$(pgrep -af 'python -m wc_simd.vlm_embed' || true)
if [ -n "$REMAINING" ]; then
  echo "[kill_vlm_embed] WARNING: Some wc_simd.vlm_embed processes still running:" >&2
  echo "$REMAINING" >&2
else
  echo "[kill_vlm_embed] All wc_simd.vlm_embed processes terminated."
fi

echo "[kill_vlm_embed] Done."
exit 0
