#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/074r3-raptor-equivalence
finish() { colab stop -s r3074 >> results/stop.log 2>&1 || true; date -u +%FT%TZ > results/stopped_at.txt; }
trap finish EXIT INT TERM
for batch in $(seq 1 200); do
 timeout --signal=KILL 660 colab exec -s r3074 --timeout 600 -f scripts/step.py > results/batch_latest.log 2>&1
 cat results/batch_latest.log >> results/batches.log
 colab download -s r3074 /content/checkpoint.tar.gz results/checkpoint.tar.gz >> results/transfers.log 2>&1
 tar xzf results/checkpoint.tar.gz -C results
 if rg -q 'Traceback|OutOfMemoryError' results/batch_latest.log; then echo 'Remote execution failed; checkpoint retained'; exit 1; fi
 echo "Saved batch $batch $(date -u +%FT%TZ)"
 if rg -q ALL_DONE results/batch_latest.log; then break; fi
done
