#!/usr/bin/env bash
# Phase 40 next-6e multi-seed sweep — 5 runs, fresh server reset each.
# Browser attacker tab MUST be running at http://localhost:8787/ntk?attack=1
# BEFORE this script starts.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/sweep
SEEDS=(1 2 3 4 5)
ROUNDS_PER_SEED=100

for S in "${SEEDS[@]}"; do
  echo "=== seed=$S ==="
  curl -s -X POST http://localhost:8787/api/ntk/reset > /dev/null
  sleep 1
  # capture moment honest-online by polling
  T_START=$(date +%s)
  ~/ntkmirror/.venv/bin/python scripts/ntk-verifier.py \
      --coord http://localhost:8787 \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --train ~/ntkmirror/examples/math_train.jsonl \
      --artifact public/data/qwen05b-math-gates-k5000.bin \
      --rounds "$ROUNDS_PER_SEED" --trials 4 \
      --device mps --dtype fp32 \
      --seed "$S" \
      --worker-id "python-honest-s$S" \
      > "logs/sweep/seed-$S.stdout" 2>&1
  T_END=$(date +%s)
  curl -s http://localhost:8787/api/ntk/state > "logs/sweep/seed-$S.state.json"
  ELAPSED=$((T_END - T_START))
  python3 - <<PY
import json
d = json.load(open("logs/sweep/seed-$S.state.json"))
ws = d.get("worker_stats") or {}
att = next((s for w,s in ws.items() if w.startswith("ntk-browser")), None)
print(f"  seed=$S  R={d.get('round')}  loss={d.get('last_loss'):.6f}  eta={d.get('eta'):.5f}  "
      f"grow={d.get('eta_grow_events')}  shrink={d.get('eta_shrink_events')}  "
      f"accept_rate={d.get('accept_rate'):.3f}  elapsed={$ELAPSED}s")
if att:
    print(f"    attacker: W={att['wins']}  F={att['frauds']}  r={att['fraud_rate']:.3f}  lwr={att['lastWinRound']}")
PY
done
echo "=== sweep complete ==="
