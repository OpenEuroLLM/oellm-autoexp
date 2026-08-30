#!/bin/bash
# Bring up (or bring back) the monitor for one session, in tmux. Run ON the cluster.
#
# Idempotent by design: safe to run when the session is missing, when the window
# is missing, when the pane is dead, and when the monitor is already healthy.
# That is the whole point -- the supervisor and a human both call exactly this,
# so a hand-launch and a supervised relaunch are the same thing.
#
#   scripts/monitor_launch.sh <session_id>
#   scripts/monitor_launch.sh --session-dir /path/to/monitor_state/<session_id>
#
# Environment:
#   AUTOEXP_TMUX_SESSION  tmux session to live in            (default: autoexp)
#   AUTOEXP_STATE_DIR     where sessions live                (default: <repo>/monitor_state)
#   AUTOEXP_ENV_SETUP     shell snippet to activate the venv (default: autodetected)
#   AUTOEXP_PYTHON        interpreter                        (default: python)
#   AUTOEXP_MONITOR_ARGS  extra args for monitor_autoexp.py  (e.g. --poll-interval 2)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION_DIR=""
SESSION_ID=""
case "${1:-}" in
  --session-dir) SESSION_DIR="${2:-}"; SESSION_ID="$(basename "$SESSION_DIR")" ;;
  "" ) echo "usage: $0 <session_id> | --session-dir <dir>" >&2; exit 2 ;;
  * ) SESSION_ID="$1"; SESSION_DIR="${AUTOEXP_STATE_DIR:-$REPO/monitor_state}/$1" ;;
esac

if [ ! -d "$SESSION_DIR" ]; then
  echo "no such session directory: $SESSION_DIR" >&2
  exit 2
fi
SESSION_DIR="$(cd "$SESSION_DIR" && pwd)"

# Presence of the stop file is a deliberate "hands off". Refusing here (rather
# than in the supervisor alone) means a stray hand-launch cannot override a stop
# either. `rm` it to resume.
if [ -f "$SESSION_DIR/.monitor.stop" ]; then
  echo "stop requested: $(cat "$SESSION_DIR/.monitor.stop")" >&2
  echo "resume with: rm $SESSION_DIR/.monitor.stop" >&2
  exit 3
fi

TMUX_SESSION="${AUTOEXP_TMUX_SESSION:-autoexp}"
WINDOW="mon-$SESSION_ID"
TARGET="$TMUX_SESSION:$WINDOW"

# Venv activation is the one genuinely site-specific bit; keep it overridable
# rather than hardcoding a JUPITER path here.
if [ -n "${AUTOEXP_ENV_SETUP:-}" ]; then
  ENV_SETUP="$AUTOEXP_ENV_SETUP"
elif [ -f "$REPO/.venv/bin/activate" ]; then
  ENV_SETUP="source '$REPO/.venv/bin/activate'"
elif [ -f "$HOME/work/venv/bin/activate" ]; then
  ENV_SETUP="source '$HOME/work/venv/bin/activate'"
else
  ENV_SETUP=":"
fi

# AUTOEXP_TMUX_TARGET is recorded in the heartbeat, so a probe (or a human) can
# find the window without guessing.
CMD="cd '$REPO' && $ENV_SETUP && export PYTHONPATH=. AUTOEXP_TMUX_TARGET='$TARGET' && \
exec ${AUTOEXP_PYTHON:-python} scripts/monitor_autoexp.py --session-dir '$SESSION_DIR' \
${AUTOEXP_MONITOR_ARGS:-}"

# remain-on-exit keeps a dead pane (and its scrollback) around, which is what
# makes `respawn-pane -k` possible and what lets you read why it died.
_arm_window() { tmux set-option -w -t "$TARGET" remain-on-exit on 2>/dev/null; }

if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  # Whole tmux server or session gone -- the login-node-reboot case.
  tmux new-session -d -s "$TMUX_SESSION" -n "$WINDOW" "$CMD" || exit 1
  _arm_window
  echo "started $TARGET (new session)"
  exit 0
fi

if ! tmux list-panes -t "$TARGET" >/dev/null 2>&1; then
  # Session survived but the window is gone.
  tmux new-window -d -t "$TMUX_SESSION" -n "$WINDOW" "$CMD" || exit 1
  _arm_window
  echo "started $TARGET (new window)"
  exit 0
fi

_arm_window
if [ "$(tmux display-message -p -t "$TARGET" '#{pane_dead}' 2>/dev/null)" = "1" ]; then
  # Pane is a corpse held by remain-on-exit: restart it IN PLACE so the window
  # keeps its position and the user's muscle memory still works.
  tmux respawn-pane -k -t "$TARGET" "$CMD" || exit 1
  echo "respawned $TARGET"
  exit 0
fi

echo "$TARGET already running"
