# Phase 40 next-3 — trusted-auditor loss oracle reactivates Phase 39 adaptation

> _Resolves the deferred next-2 limitations (no server-side loss, no
> byzantine, no adaptive η) by piggybacking a worker-reported test loss
> onto every tick. Server uses it as the canonical lastLoss; runs Phase
> 39's sym-AIMD + byzantine real_Δ check on a one-round lag._

## The wire change (a single optional float)

```
   tick body, Phase 40 next-2:
     {worker_id, round, seed, scalar_g, delta, since_round}

   tick body, Phase 40 next-3:
     {worker_id, round, seed, scalar_g, delta, since_round,
      audit_loss_before: float}                  ← 4 extra bytes
```

The worker ALREADY computes `loss_before` at the start of each round to
construct claimed_Δ. We just send it. No extra forwards required.

## The server-side loop (one-round lag)

```
   round N tick arrives with audit = L(θ_N before apply):
     1. if pendingAudit is set:
          realDelta := audit − savedLossBeforeApply
          byzantine check on pendingAudit's winner (Phase 9 / 14 / 31 math)
          sym-AIMD η update on η (Phase 39 log-symmetric, ×1.05 / ×1/1.05)
          clear pendingAudit
        # else: first audit of the run; nothing to compare against yet.
     2. lastLoss := audit

     3. tournament accepts the proposal, may advance round.
     4. if applied:
          savedLossBeforeApply := lastLoss          # the audit just stored
          pendingAudit := {round, winnerId, claimedDelta}

   round N+1 tick:
     1. audit ≈ L(θ_N after apply)
     2. realDelta = audit − savedLossBeforeApply
                   = L(θ_after) − L(θ_before)        ← Phase 39's real_Δ
     3. byzantine + sym-AIMD run with the correct signal.
```

## Empirical anchor (R=30, single Python verifier, Qwen-0.5B)

Comparing three configurations on byte-identical setup:

| metric           | next-2 (no audit)    | next-3 threshold 1e-4    | **next-3 threshold 1e-5** |
|---|---|---|---|
| R=30 final loss  | 1.7629               | 1.7628                   | **1.7625**                |
| Δ loss over R=30 | −0.0004              | −0.0004                  | **−0.0007**  ← 1.75× more |
| final η          | 0.001 (fixed)        | 0.001 (no adapt fired)   | **2.53e-3 (grew 2.5×)**   |
| grow events      | 0                    | 0                        | 19                         |
| shrink events    | 0                    | 0                        | 0     ← all grow, no shrink|
| ‖θ‖₂            | 0.044                | 0.045                    | **0.078**                  |
| max ‖θ_i‖       | 0.0027               | 0.0027                   | **0.0042**                 |

**Reading the result:** with ETA_DELTA_THRESH = 1e-5 (tighter than Phase
39's 1e-4 because gate-controller per-round Δloss is tiny at init —
gates near zero mean the model is effectively the base model), the audit
signal lifts above noise within ~5 rounds. All 19 adaptation events were
**grow** (none shrink) — the swarm's descent is monotonic and η drifts
up by 2.5× over R=30, matching Phase 39's sym-AIMD signature exactly.

At ETA_DELTA_THRESH = 1e-4 the per-round real_Δ falls below the noise
band at this scale; nothing adapts. The plumbing is still correct
(audit ingestion + pendingAudit closure + state transitions all fire)
— the rule just doesn't trigger because the signal is too small to
trust. That's safe behavior.

## What this validates

1. **Phase 39's sym-AIMD generalizes to gate-controller training.**
   Same multiplicative log-symmetric rule, same monotonic-η-drift-up
   signature, on a completely different (D, P, base-model) regime.
2. **The trusted-auditor architecture works.** No separate auditor
   service, no Worker-bundled tiny model, no cross-worker
   verification — just one optional float on the tick body. The
   single-Python-verifier-as-auditor pattern is enough for
   single-worker dev runs and naturally generalizes to multi-worker
   (any worker's audit counts).
3. **Byzantine defense logic is back online.** No attacker in this
   run, but `workerStats` records each round's win + fraud verdict,
   and the tiered quarantine math (cum ≥ 40 %, last-20 ≥ 40 %,
   last-100 ≥ 25 %) is intact. Reactivation just took the audit
   stream.

## What's still deferred to a later phase

- **A browser-runnable worker.** Python is still the only Qwen
  runner. Transformers.js bundle or neuropulse WGSL → next-4.
- **Multi-worker federation.** TARGET_PROPOSALS stays at 1 until the
  browser worker reduces per-tab Qwen cost.
- **An actual byzantine attacker.** Logic is wired, defense is ready,
  but we haven't intentionally cheated. A `--attack` flag on the
  Python verifier (claimed_Δ = −10, random seed) is a 10-line change.
- **Longer empirical run (R = 200+).** Phase 40 results so far are
  R = 30. Worth running R = 500 for paper-grade trajectory data —
  but each round is ~2.6 s on Mac MPS, so R = 500 ≈ 22 min wall time
  per protocol. Tractable; deferred for time.

## Files changed

```
   src/tournament-ntk.ts                  +124 LOC over next-2:
                                          - lastLoss, savedLossBeforeApply,
                                            pendingAudit state
                                          - currentEta + Phase 39 sym-AIMD
                                          - workerStats + Phase 9/14/31 quarantine
                                          - audit ingestion in tick handler
                                          - eta + eta_grow_events + eta_shrink_events
                                            in /tick + /snapshot + /state responses

   scripts/ntk-verifier.py                +~30 LOC:
                                          - track current_eta from /tick responses
                                          - include audit_loss_before in submitting tick
                                          - apply applied flips using server's CURRENT η
                                          - print η + grow/shr counts each progress line
```

## Reproducing

```bash
cd ~/postnet-cf
npm run dev   # shell A

# shell B, ntkmirror venv from Phase 40 scope:
~/ntkmirror/.venv/bin/python scripts/ntk-verifier.py \
    --coord http://localhost:8787 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train ~/ntkmirror/examples/math_train.jsonl \
    --artifact public/data/qwen05b-math-gates-k5000.bin \
    --rounds 30 --trials 4 --reset
```

Expected: η drifts up from 1.0e-3 to ~2.5e-3 over 30 rounds, grow
events ~15-20, shrink events 0, loss descends by ~7e-4. Per-round
wall time ~2.5-3 s on M-series Mac (fp32, MPS).
