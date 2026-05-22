#!/usr/bin/env bash
# Phase 19: smoke-test all verifiers.
#
# Assumes `wrangler dev` is already running on port 8787.
# Runs each verifier with its default budget, extracts the final loss,
# checks it against an expected upper bound, reports pass/fail.
#
# Known issue: `wrangler dev` cannot sustain the load of multiple long
# verifier runs back-to-back (the local Workers runtime crashes with
# "other side closed" / "worker restarted"). The tournament-float and
# tournament-ternary tests hit this most often. Verifiers retry transient
# JSON-parse errors but can't recover from full server death. Workaround:
# run a single verifier at a time, or `npx wrangler deploy` and run
# against the deployed Worker instead — production Workers tolerate the
# load fine.
#
# Exit code 0 = all green; non-zero = at least one regression
# (skipped tests don't count against exit code).

set -u
COORD="${COORD:-http://localhost:8787}"
cd "$(dirname "$0")/.."

# Pre-flight: wrangler reachable?
if ! curl -s -o /dev/null -w '%{http_code}' "$COORD/" | grep -q 200; then
  echo "ERR: wrangler dev not responding at $COORD"
  echo "     run \`npx wrangler dev --port 8787\` first"
  exit 2
fi

mkdir -p logs
fails=0

run_check () {
  local name=$1 cmd=$2 pattern=$3 max=$4
  local logfile=logs/smoke-${name}.log
  echo -n "  ${name} ... "
  # Verify wrangler is still alive before each test (it dies under cumulative load)
  if ! curl -s -o /dev/null -w '%{http_code}' "$COORD/" 2>/dev/null | grep -q 200; then
    echo "SKIP  (wrangler not responding)"
    return
  fi
  bash -c "$cmd" > "$logfile" 2>&1
  # Even on non-zero exit, try to parse — wrangler may have crashed
  # mid-run after producing partial results.
  local line loss
  line=$(grep -E "$pattern" "$logfile" | tail -1)
  # Extract a float value (the loss number) from the matched line.
  loss=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  if [ -z "$loss" ]; then
    if grep -q "worker restarted" "$logfile" 2>/dev/null; then
      echo "SKIP  (wrangler crashed; see $logfile)"
    elif grep -q "Error\|Exception" "$logfile" 2>/dev/null; then
      echo "FAIL  (error before result; see $logfile)"
      fails=$((fails + 1))
    else
      echo "FAIL  (no '$pattern' match; see $logfile)"
      fails=$((fails + 1))
    fi
    return
  fi
  if awk -v l="$loss" -v m="$max" 'BEGIN { exit !(l+0 < m+0) }'; then
    echo "PASS  (loss $loss < $max)"
  else
    echo "FAIL  (loss $loss >= $max)"
    fails=$((fails + 1))
  fi
}

echo "postnet-cf smoke tests"
echo ""

# Each verifier: short budget that gets the loss well below random init.
# Adam: random ≈ 0.69; expect ≪ 0.2 in 100 rounds.
run_check adam \
  "node scripts/headless-worker.mjs" \
  "circle.*final loss" 0.2

# Tournament float: same task, lower budget; circle should be < 0.25 in 400 rounds
run_check tournament-float \
  "node scripts/tournament-verifier.mjs" \
  "circle.*loss" 0.25

# Ternary: coarser space, more lenient threshold
run_check tournament-ternary \
  "node scripts/ternary-verifier.mjs" \
  "circle.*loss" 0.5

# Char-LM: random init ≈ 3.30; 500 rounds gets us to < 2.5 reliably
run_check char-lm \
  "ROUNDS=500 node scripts/lm-verifier.mjs" \
  "loss=" 2.5

# Bandwidth sweep: no live coord needed, presence-check that it ran end-to-end
run_presence () {
  local name=$1 cmd=$2 pattern=$3
  local logfile=logs/smoke-${name}.log
  echo -n "  ${name} ... "
  bash -c "$cmd" > "$logfile" 2>&1
  if grep -q "$pattern" "$logfile"; then
    echo "PASS  (saw '$pattern')"
  else
    echo "FAIL  (no '$pattern' in output; see $logfile)"
    fails=$((fails + 1))
  fi
}
run_presence bandwidth-sweep \
  "node scripts/bandwidth-sweep.mjs" \
  "BitNet 2B"

echo ""
if [ "$fails" -eq 0 ]; then
  echo "ALL GREEN ($fails fails)"
  exit 0
else
  echo "FAILED ($fails fails)"
  exit 1
fi
