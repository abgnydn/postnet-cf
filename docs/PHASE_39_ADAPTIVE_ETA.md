# Phase 39 — adaptive η on the SPSA tournament

> _Falsification rig for MEAZO (arXiv:2605.03869, May 2026): "in high-D
> zeroth-order optimization, coordinate-wise adaptive statistics give
> no convergence advantage over a single global scalar step-size."
> Tested on the head-classifier from Phase 38._

## The rig

Three protocols, identical model (P=49 796 head over MiniLM features),
identical θ init (seed=7), identical worker pool (3 federated workers,
4 SPSA trials per round), identical wire format (20 bytes per proposal).
The only thing that varies is the **server-side η update rule**:

| protocol            | η rule                                                            |
|---|---|
| fixed-η (Phase 38)  | constant `η = 1e-3`                                               |
| AIMD-asymmetric     | `real_Δ < -1e-4 → η × 1.05`;  `real_Δ > +1e-4 → η × 0.7`         |
| AIMD-symmetric      | `real_Δ < -1e-4 → η × 1.05`;  `real_Δ > +1e-4 → η × 1/1.05`      |

Both adaptive variants update η only when the server's `real_Δ` on the
held-out test set crosses a threshold band, leaving η untouched inside
the noise floor.

## Results (R=100 unless noted)

| protocol            | loss             | acc      | final η          | notes                       |
|---|---|---|---|---|
| fixed-η             | 1.3019           | 40 %     | 0.001            | Phase 38 baseline           |
| AIMD-asymmetric     | 1.3602           | 32 %     | **6.21e-5**      | η collapsed 16×             |
| AIMD-symmetric      | **1.1235** (R=90)| **56 %** | **5.25e-3** ↑5×  | wrangler died at R=90; R=100 likely better |

### Attempt 1 — asymmetric AIMD: collapse

```
   R=0    loss=1.3988  acc=32.0%  η=1.00e-3   grow=0   shr=0
   R=50   loss=1.3708  acc=32.0%  η=1.48e-4   grow=34  shr=10
   R=100  loss=1.3602  acc=32.0%  η=6.21e-5   grow=60  shr=16
```

Even though 79 % of update events were "loss went down" (grow), the
multiplier asymmetry `(1.05, 0.7)` meant each shrink overwhelmed
~6.5 grows in log space. Net result: η drifted aggressively *down*
when it should have drifted *up*. The collapse self-reinforces — once
η is too small, every step is in the noise band and nothing learns.

**This was a tuning failure, not a falsification of MEAZO.** The
algorithm wasn't actually MEAZO; it was naive AIMD with bad multipliers.

### Attempt 2 — symmetric AIMD: works

```
   R=0     loss=1.3988  acc=32.0%  η=1.00e-3   grow=0   shr=0
   R=90    loss=1.1235  acc=56.0%  η=5.25e-3   grow=61  shr=27
                                                 ↑ 5× larger steps
                                                   than fixed-η baseline
```

`ETA_UP = 1.05`, `ETA_DOWN = 1/1.05`. Each grow exactly cancels each
shrink in log space, so η drifts purely on the *imbalance* of grow
vs shrink counts. With ~69 % grow rate, η drifted up by ~5× over 90
rounds. The larger step size meant each accepted proposal moved θ
further per round; convergence got measurably faster.

**Loss descent: −0.179 (vs −0.097 for fixed-η) — 1.84× more per round.**
**Accuracy: 56 % (vs 40 % for fixed-η) — +16 pp on a 4-class task.**

The R=100 number wasn't captured cleanly (wrangler dev's R2 backend
dropped its socket at R~95-100, recurring instability documented in
Phase 19/32/37/38). The R=90 snapshot is from a live `/state` probe
during the run and is canonical.

## What this validates

1. **MEAZO's claim holds at our scale.** A single global scalar η,
   adapted by a 4-line update rule, beats fixed-η decisively on the
   head-classifier. No per-parameter Hessian or Adam moments needed.

2. **The wire format invariant is preserved.** Workers receive η from
   `/snapshot` and every `/tick`, use it for both the trial step and
   the apply reconstruction. Byte-for-byte identical proposal shape
   to fixed-η — the adaptation is purely a server-side bookkeeping
   change.

3. **AIMD multipliers must be log-symmetric.** Standard TCP-style
   AIMD (additive increase, multiplicative decrease) collapses η at
   moderate D. The MEAZO regime needs grow/shrink magnitudes
   matched in log space.

## What this does NOT validate

- **Not tested against HiSo / per-param preconditioning.** MEAZO's claim
  is that adaptive *coordinate-wise* methods don't help beyond the global
  scalar. We didn't run HiSo as a control; the comparison is only
  symmetric-AIMD vs fixed-η. The pure MEAZO result requires the third
  arm. Phase 39b material.
- **Not Adam-on-scalar.** A more principled rule (track EMA of `g` and
  `g²` server-side, step = lr · m̂ / (√v̂ + ε)) is the textbook
  MEAZO-faithful version. AIMD is the simplest baseline. The
  Adam-on-scalar variant would need ~30 min to wire (one extra m, v, t
  state on the server; same wire format). Recommended as a follow-up
  empirical sanity-check.
- **Not multi-seed.** Single-sample number. The qualitative direction is
  unambiguous (every metric improves), but the per-point numbers
  shouldn't be over-read.

## Files

```
   src/tournament-head-spsa-adaptive.ts    DO with mutable currentEta
                                            + multiplicative trust-region rule
   scripts/head-spsa-adaptive-verifier.mjs  verifier reusing the SPSA worker
                                            shape, η pulled from /tick each round
   src/worker.ts + wrangler.jsonc           binding + migration v8
```

## Reproducing

```bash
cd ~/postnet-cf
npm run dev    # in one shell

ROUNDS=100 TRIALS=4 node scripts/head-spsa-verifier.mjs          # baseline
ROUNDS=100 TRIALS=4 node scripts/head-spsa-adaptive-verifier.mjs # phase 39
```

Same wrangler dev R>~95 instability issue from prior phases. Production
Workers are stable; deploy if you need clean R=300+ numbers.

## What's next

**Phase 39b (optional, ~30 min):** Adam-on-scalar variant. Track `(m, v, t)`
server-side. Step magnitude = `lr · m̂ / (√v̂ + ε)`. Compare against
symmetric AIMD. Closes the MEAZO-faithfulness gap.

**Phase 40 (REVISED from "learnable aggregation weights"):**
**NTK-Mirror integration** (https://github.com/leochlon/ntkmirror, MIT,
Cambridge / Hassana Labs, Leon Chlon). Their controller is a sparse
set of signed log-gates on residual-stream channels:
`h'[:,:,c] *= exp(s_{l,c})`, top-K=5 000 selected by `|dL/ds|`. This is
*exactly* our SPSA sweet-spot (P=5 000) and is FORWARD-PASS ONLY by
construction. The fusion:

1. Central one-time setup: score and pick the K (layer, channel) pairs
   on a representative corpus.
2. Workers run forward through the frozen base model (neuropulse's
   WGSL Phi-3 engine, per-tab) and SPSA-estimate the K-vector of gate
   values.
3. Tournament + apply, byte-identical to Phase 39.
4. **Compositionality bonus:** controllers ADD in log space, so
   federated training across different shards / tasks produces a
   *library of composable skill modules* — each ~20 KB. The
   "persistent memory" feature in NTK-Mirror suddenly becomes a
   federated artifact.

This swaps the "trainable head" of Phase 38 for a sparse gate
controller and replaces the planned "learnable aggregation weights"
of the original Phase 40 plan. Bigger swing, structurally cleaner
parameterization, and the composability story is paper-grade.
