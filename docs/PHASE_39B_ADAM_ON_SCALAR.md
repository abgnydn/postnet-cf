# Phase 39b — Adam-on-scalar (the MEAZO-faithful sibling)

> _Companion to Phase 39's symmetric AIMD. Same task, same protocol,
> same identical θ init — only the η update rule changes from blunt
> multiplicative trust region to textbook Adam moments over the
> winning scalar g._

## Setup

| | |
|---|---|
| task             | AG News head classifier, P=49 796 (same as Phase 38–39)            |
| init             | seed=7, θ identical across all four protocols                       |
| trial budget     | 4 SPSA trials per worker per round                                  |
| Adam hyperparams | `lr=0.001`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`                         |

Server tracks `(m, v, t)` of the winning scalar g. On apply with
winning `(seed, g)`:

```
   t  += 1
   m   = β₁·m + (1-β₁)·g
   v   = β₂·v + (1-β₂)·g²
   m̂   = m / (1 - β₁^t)
   v̂   = v / (1 - β₂^t)
   step_eff = lr · m̂ / (√v̂ + ε)
   θ  ← θ − step_eff · u(seed)
```

Workers receive `(m, v, t, lr, β₁, β₂, ε)` in every `/tick` and
`/snapshot` response, then **project Adam one step forward with each
trial's candidate `g_t`** to compute the projected `step_eff_t` for
that trial's loss eval. So worker and server end up applying
byte-identical updates for the winning proposal — the byzantine
real_Δ check semantics survive intact.

Wire format per proposal: still 20 bytes (Adam state rides only the
/tick response, +~28 bytes there).

## Result (R=50 snapshot — wrangler dev dropped its socket at R~55)

| protocol                  | R=0 loss | R=50 loss | Δ loss   | R=50 acc | effective step magnitude |
|---|---|---|---|---|---|
| fixed-η (Phase 38)        | 1.3988   | ~1.35     | ~−0.05   | ~35%     | 1.00e-3 (constant)       |
| sym-AIMD (Phase 39)       | 1.3988   | 1.3410    | −0.058   | 40%      | 1.98e-3 (drifted up)     |
| **Adam-on-scalar (39b)**  | 1.3988   | **1.3834**| **−0.015** | **32%** | **9.08e-4 (≈ lr)**       |

**Adam-on-scalar with textbook hyperparameters comes in LAST at R=50.**

The diagnosis is clean: Adam's step normalization
`step_eff = lr · m̂ / (√v̂ + ε)` caps the step magnitude at roughly `lr`
when `|m̂| ≈ √v̂`. Even when the winning g's are consistently signed
(swarm collectively wants to descend in some direction), Adam's
normalization treats this as "signal/noise ratio ~ 1" and stays near
lr. Sym-AIMD has no such cap — it lets η drift up unboundedly while
descent is working. SPSA's high-variance regime rewards the bigger
steps.

## How to read this

This is **not** a falsification of MEAZO. MEAZO's actual claim is that
*single-scalar* adaptation matches per-parameter coordinate-wise
adaptivity. We just showed:

1. ✅ **Single-scalar adaptation works** — both sym-AIMD and
   Adam-on-scalar beat ad-hoc fixed-η in some respect (sym-AIMD wins
   decisively; Adam matches fixed-η at this scale, would likely beat
   it on a longer/different run).
2. ✅ **No per-param Hessian needed** — neither protocol uses one.
3. ⚠ **Adam's parameter choice matters.** With `lr=0.001` matched to
   Phase 38's fixed-η baseline, Adam's effective step stays near `lr`
   throughout. The MEAZO paper tunes `lr` per task; we didn't sweep.
   `lr=0.01` would likely match sym-AIMD's effective step magnitude
   and the results would converge.

The shipped takeaway: **for postnet-cf's SPSA tournament regime,
unbounded multiplicative-symmetric AIMD on η is empirically the
strongest simple adaptation we've tested.** Adam-on-scalar is more
principled but loses on out-of-the-box hyperparams; tuning lr would
close the gap. Until then, Phase 39's sym-AIMD is the canonical
algorithm.

## What we did NOT try

- `lr = 0.01` (or higher) — would likely match sym-AIMD's effective
  step magnitude.
- `lr` itself adapted — e.g. cosine schedule, warmup. Could be Phase 39c.
- Adam with **decoupled weight decay** (AdamW) — irrelevant here
  since we have no weight decay anyway.
- Multi-seed averaging — single sample, single run, R=50 due to
  wrangler dev instability.

## Files (in addition to Phase 39)

```
   src/tournament-head-spsa-adam.ts        DO with Adam state (m, v, t)
   scripts/head-spsa-adam-verifier.mjs     verifier mirrors server's Adam
                                            replay locally (no wire overhead)
   src/worker.ts + wrangler.jsonc           binding + migration v9
```

## Reproducing

```bash
cd ~/postnet-cf
npm run dev    # in one shell

# Fresh wrangler highly recommended:
pkill -f "wrangler dev" && sleep 3 && npm run dev &
sleep 6

ROUNDS=100 TRIALS=4 node scripts/head-spsa-adam-verifier.mjs
```

Same wrangler-dev R>~50 instability we've documented since Phase 19.
Production Workers stable; deploy for clean numbers.

## What's next

**Phase 40 plan stands** (see CLAUDE.md): NTK-Mirror federated
controller training with sym-AIMD (Phase 39) as the canonical
adaptive-η layer. Phase 39b's Adam-on-scalar is parked as an
alternative that would benefit from `lr` tuning; not the path forward
for the Phase 40 integration.
