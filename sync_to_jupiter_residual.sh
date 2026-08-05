#!/bin/bash
# Isolated sync target for residual-architecture speed experiments.
#
# The main sync writes into ~/work/Projects/oellm-autoexp, which is the directory
# the in-flight science runs launched from. Already-running jobs are safe (their
# Python is imported into memory), but a requeued or restarted job would pick up
# half-synced code. Speed experiments iterate fast and would otherwise collide,
# so they get their own tree.
rsync -ru --progress --exclude monitor_state --exclude *.sif ./ jupiter:~/work/Projects/oellm-autoexp-residual/
