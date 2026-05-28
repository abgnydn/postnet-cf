# Phase 40 next-6 empirical — byzantine defense under live attack

> _Validation run for the no-self-audit + magnitude-lie fixes shipped
> in commit `4ff5190`. One honest Python verifier (Qwen-0.5B on Mac MPS)
> + one browser attacker tab (`?attack=1`, fabricates `claimed_delta = -10`
> every round, no forward pass)._

## Setup

| component | role     | speed                   | notes                                        |
|---|---|---|---|
| `scripts/ntk-verifier.py` | honest worker | ~5 s/round on MPS | full SPSA, sends `audit_loss_before`         |
| Chrome tab `/ntk?attack=1` | attacker      | ~0.5 s/round         | skips ONNX entirely; spams seed + claim=-10  |
| local `wrangler dev`      | coordinator   |                      | NTK DO, `TARGET_PROPOSALS = 2`               |

Same machine (M-series Mac). Attacker is up first; Python verifier
starts ~15 s later (Qwen load + MPS warmup).

## Result — attacker quarantined at W=10

The byzantine signal hits 100 % from the first audit.

| t (s) | server R | honest audits | attacker stats         | note                                  |
|---|---|---|---|---|
|  60   | 32       | 0 → 1 (first) | W=2 F=2 r=1.00         | first 6 attacker wins enqueued before honest online; first audit closes the oldest |
|  75   | 39       | +2            | W=2 F=2 r=1.00         | magnitude-lie test on every closure   |
|  90   | 47       | +1            | W=3 F=3 r=1.00         |                                       |
| 105   | 52       | +1            | W=4 F=4 r=1.00         |                                       |
| 120   | 55       | +2            | W=6 F=6 r=1.00         |                                       |
| 135   | 62       | +2            | W=8 F=8 r=1.00         |                                       |
| **150** | **64** | **+2**        | **W=10 F=10 r=1.00**   | **quarantine threshold hit (cumRate > 0.4 with wins ≥ 10)** |

That's **~90 s of audit time** from "honest worker online" to
"attacker kicked out." On a coordinator with many honest peers and
a single bad actor, the kick would be substantially faster because
each audit lands across distinct workers and the queue drains
proportionally to the auditor count.

## Why every audit flagged

The attacker's `claimed_delta = -10` is a hard lie about magnitude.
Honest per-round Δloss for K = 5 000 gates at η ≈ 1e-3 is `O(1e-3)`.
After server applies the attacker's fabricated seed with η · g · u
where g = 1.0 and u has unit perturbations, the real loss change is
`|Δ| < 0.02`. The Phase 40 next-6 magnitude test:

```ts
const magnitudeFraud = claimedDelta < (realGlobalDelta - 0.5);
//                     -10           < (    ±0.02       - 0.5)
//                     -10           < -0.48
//                                = true (always)
```

trips on every single audit. The Phase 39 inversion test (real Δ > 0
when claimed < 0) only fires ~50 % of the time at this perturbation
scale because random direction lowers loss half the time — without the
magnitude check, the cumulative rate would stall around 0.5 and never
hit the 0.4 threshold within 10 wins. Magnitude was the missing piece.

## Post-quarantine — η climbs, loss steadily descends

Once the attacker is quarantined, the python verifier drives the
swarm. With Phase 39 sym-AIMD and the magnitude-fraud check
permitting almost all of honest's `realDelta < -1e-5` events to
fire ETA_UP, η climbs monotonically and the loss curve becomes
linear-ish in round count:

| server R | last_loss | Δ vs baseline | η         | grow events |
|---|---|---|---|---|
| 64       | 1.763168  | —             | 0.00100   | 0           |
| 75       | 1.762816  | −0.00035      | 0.00279   | 21          |
| 100      | 1.761660  | −0.00151      | 0.00279   | 21          |
| 130      | 1.760304  | −0.00286      | 0.00279   | 21          |
| 161      | 1.758480  | −0.00469      | 0.00279   | 21          |
| 183      | **1.756951** | **−0.00622**  | 0.00279   | 21 (0 shrink) |

The descent rate stabilises around **−3.5 × 10⁻⁵ per round** once
η has reached its self-organised plateau. No shrink events ever
fire — every single accepted apply lowers honest's measured loss by
more than `ETA_DELTA_THRESH = 1e-5`.

Final coordinator state at server R=183:

```
last_loss          1.7569507
eta                0.00279 (2.79× initial)
eta_grow_events    21
eta_shrink_events  0
accepted           183 / 367 considered (49.9% accept rate)
pending_audits     64 (capped — see "design limit" below)
worker_stats       attacker: W=33 F=33 fraud_rate=1.000
theta_stats        l2_norm=0.634  mean_abs=0.0071  max_abs=0.034
```

The K=5000 gate controller has measurably evolved (l2_norm 0 → 0.63;
gate values now span roughly [-0.034, +0.034] in log-space). All of
this came across the 20-byte wire format with the attacker active
the entire run.

## Design limit surfaced: solo-honest auditing pauses

With only ONE honest worker post-quarantine, the no-self-audit rule
(Phase 40 next-6 fix #1) prevents the lone honest peer from closing
its own pending entries either. After ~64 honest wins, the queue
hits cap and oldest entries start evicting silently. **The system
still progresses** — applies happen, loss descends, η adapts — but
the byzantine signal goes quiet because there is no one to audit
the only honest worker. This is the correct behaviour, not a bug:
in a single-honest swarm there is no second opinion to leverage.
Production deployments with ≥ 2 honest workers don't see this; the
queue stays drained.

## What the run validates

1. **No-self-audit rule works.** The attacker's submit-payload no
   longer carries `audit_loss_before`, and the server-side
   `findIndex(p => p.workerId !== body.worker_id)` correctly
   refuses to let the attacker close its own pending entries.
   Without this, the magnitude check would never have audit data
   to operate on.

2. **`audit_loss_before > 0` guard prevents corruption.** During
   the run, `last_loss` stays at honest python's actual cross-
   entropy value the whole time — no sentinel zeros sneak through.

3. **Magnitude-lie fraud test catches the "lucky downhill" attack.**
   Pre-magnitude, on the same setup, the attacker accumulated 60+
   wins with fraud rate ~1.5 % (random walk happened to lower loss
   for the majority of applies; inversion test missed them). Post-
   magnitude, every audit fires; quarantine is reached in 10 wins.

4. **Phase 14/31 quarantine thresholds (cumRate > 0.4) fire
   cleanly.** The check `wins ≥ 10 AND fraud_rate > 0.4` activates
   the moment the 10th win is recorded. The attacker's next
   proposal is rejected with `quarantined: true`.

## Reproducing

```bash
# terminal 1 — coordinator
cd ~/postnet-cf && npm run dev

# terminal 2 — onnx asset server (for the honest browser tab path; not
#              used in this run but kept up for the demo URL)
cd ~/postnet-cf-onnx && npx http-server -p 8788 --cors -c-1 .

# reset coord
curl -s -X POST http://localhost:8787/api/ntk/reset

# Chrome tab — attacker
open "http://localhost:8787/ntk?attack=1"
# click Join

# terminal 3 — honest worker (~5 s/round on M-series Mac, MPS)
~/ntkmirror/.venv/bin/python ~/postnet-cf/scripts/ntk-verifier.py \
    --coord http://localhost:8787 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train ~/ntkmirror/examples/math_train.jsonl \
    --artifact ~/postnet-cf/public/data/qwen05b-math-gates-k5000.bin \
    --rounds 250 --trials 4 \
    --device mps --dtype fp32 \
    --worker-id python-honest

# observe — server state
watch -n 5 "curl -s http://localhost:8787/api/ntk/state | python3 -c '
import sys,json; d=json.load(sys.stdin)
print(f\"R{d[chr(34)+chr(114)+chr(111)+chr(117)+chr(110)+chr(100)+chr(34)]} loss={d[chr(34)+chr(108)+chr(97)+chr(115)+chr(116)+chr(95)+chr(108)+chr(111)+chr(115)+chr(115)+chr(34)]:.6f}\")
'"
```

(use any state-poll script; `scripts/ntk-empirical-monitor.sh` is the
quick one used here.)
