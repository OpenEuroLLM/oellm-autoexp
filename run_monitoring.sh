#!/bin/bash

session_id=$1 # 1770027288

source use_env.sh

PYTHONPATH=. python scripts/monitor_autoexp.py   --session-dir ./monitor_state/$session_id
