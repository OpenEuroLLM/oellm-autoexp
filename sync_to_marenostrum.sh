#!/bin/bash
rsync -ru --progress --exclude monitor_state --exclude *.sif ./ marenostrum:~/work/Projects/oellm-autoexp/
