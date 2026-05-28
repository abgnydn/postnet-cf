# Phase 40 next-2 — federated NTK-Mirror controller training

> _What this session shipped: the federated DO + a Python verifier that
> drives real Qwen-0.5B forward passes through our gate-apply hook.
> Validates the protocol layer end-to-end on a real LLM. Loss-oracle +
> byzantine defense are explicitly deferred to next-3._

## What's wired up

```
   ┌──────────────────────────────────────────────────────────────┐
   │ public/data/qwen05b-math-gates-k5000.bin     (Phase 40 next-1)│
   │   (40 KB static artifact — gate indices + raw_init)          │
   └─────────┬────────────────────────────────────────────────────┘
             │ fetched via env.ASSETS on construction
             ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ TournamentNtk (src/tournament-ntk.ts) — Cloudflare Worker DO │
   │                                                                │
   │   theta  = raw[K=5000]  (trainable scalars, init = zeros)     │
   │   /api/ntk/tick     — same shape as Phase 36+ SPSA tournament │
   │   /api/ntk/snapshot — R2-sharded θ snapshot                    │
   │   /api/ntk/ws       — broadcast applied flips                  │
   │                                                                │
   │   apply: θ ← θ − η · scalar_g · u(seed)        (fixed η=0.001) │
   │   wire: (seed, scalar_g, claimed_Δ) = 20 bytes per proposal    │
   │                                                                │
   │   ✗ NO server-side test loss (CF Worker can't load Qwen-0.5B) │
   │   ✗ NO byzantine real_Δ check (no server loss to compare to)  │
   │   ✗ NO adaptive η (Phase 39's sym-AIMD needs server loss)     │
   └─────────┬────────────────────────────────────────────────────┘
             │
             ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ scripts/ntk-verifier.py  (one Python process = one worker)   │
   │                                                                │
   │   loads Qwen-0.5B (~1 GB) via Hugging Face Transformers       │
   │   loads gate artifact (same parser as TS, byte-identical)     │
   │   attaches ntkmirror's _SignedLogMaskModule with our raw      │
   │   per round:                                                   │
   │     polls /tick, reconciles applied_since                     │
   │     T=4 SPSA trials:                                          │
   │       seed_t = u32 random                                     │
   │       u_t = mulberry32+Box-Muller(seed_t)  ← matches TS       │
   │       loss_plus  = forward(θ + ε·u)                            │
   │       loss_minus = forward(θ − ε·u)                            │
   │       g_t = (loss_plus − loss_minus) / 2ε                     │
   │       loss_at = forward(θ − η·g_t·u)                          │
   │       claimed_Δ_t = loss_at − loss_before                     │
   │     submits the best trial                                    │
   └──────────────────────────────────────────────────────────────┘
```

## Empirical anchor (R=30, single Python worker, M-series Mac)

```
   start loss:    1.7632   ← model at raw=0 ≡ base model unchanged
   final loss:    1.7629   ← 30 rounds later, gates non-zero
   Δ loss:       −0.0004   ← monotonic descent over the run

   loss trajectory (per ten rounds):
     R= 0   1.7632
     R=10   1.7631
     R=20   1.7630
     R=30   1.7629

   ||θ||₂:        0 → 0.0441    (theta evolved off zero)
   max |θ_i|:    0 → 0.0023    (gates using ~5% of the 0.05 budget)
   per-round time: ~1.3 sec on MPS, fp32, batch=4, max_length=256
```

The descent is small per round because gates start at zero — the
SPSA gradient signal at θ=0 is weak (the model's behavior barely changes
across the perturbation). As gates grow, the per-round Δ should compound.
Phase 40 next-3 will run longer (R=200+) and quantify the trajectory.

## What this validates

1. **The full federated loop runs on a real LLM.** Python process
   submits proposals → Cloudflare Worker DO applies SPSA updates →
   subsequent forwards use the updated gates → loss descends. End to end.
2. **The binary artifact pipeline works.** Python writer
   (`scripts/extract-ntk-gates.py`, Phase 40 next-1) → TS reader
   (`src/ntk-gate.ts`) → Python re-reader (this verifier) are all
   byte-compatible. FNV-1a 64 model hash matches across all three.
3. **The SPSA perturbation is bit-reproducible across TS ↔ Python.**
   Same `mulberry32 + Box-Muller` scheme used since Phase 36 carries
   over. Workers and the DO compute byte-identical u from any seed.

## What this deferred (next-3 targets)

| capability                       | why deferred                                                            |
|---|---|
| server-side test-loss            | CF Worker can't load Qwen-0.5B (~500M params, ~1 GB weights)           |
| post-apply byzantine check       | needs server-side test-loss to compute `real_Δ` vs `claimed_Δ`         |
| adaptive η (Phase 39 sym-AIMD)   | needs server-side `real_Δ` signal to drive the AIMD growth/shrink rule |
| TARGET_PROPOSALS = 2             | with one Python verifier (heavy on RAM), one is the only practical N    |
| browser-tab worker               | Python is the only Qwen runner we have today; next-3 adds a JS/WGSL one |

Phase 40 next-3's job: bring back ALL of the above by adding a
loss-oracle path that the server can call. Two viable architectures:

  a. **A "trusted auditor" companion service** that runs Qwen forward
     outside the Worker, posts test_loss to the DO via a privileged
     endpoint per round. Easy. Adds an off-Worker dependency.

  b. **A small enough oracle that DOES fit in a Worker** — e.g.,
     Transformers.js bundled into the Worker for an even tinier base
     model (Pythia-160M, DistilGPT-2). Phase 39's full Adaptive-η +
     byzantine defense reactivates. Self-contained.

  c. **Cross-worker verification** — server doesn't compute test
     loss but asks ANOTHER worker to evaluate the post-apply theta;
     fraud = disagreement. Decentralized. Most novel.

(c) is the most interesting research direction but adds protocol
complexity. (a) is the simplest path to "Phase 40 next-3 ships".

## Memory note (for repro)

The Phase 40 next-2 verifier uses ~3-4 GB of RAM (Qwen-0.5B fp32
weights + activations + Python overhead + transformers cache). On a
16 GB Mac this is tight when wrangler dev + a browser + Claude Code
are also running. If you hit OOM pressure:

```
  --dtype bf16        # ~half memory; risk: NaN at training step 8 on MPS
  --device cpu        # slower, no GPU memory pressure
  --batch-size 2      # half activation memory
  --max-length 128    # half tokens per batch
```

Or move to a smaller base model (e.g. Pythia-160M, ~640 MB fp32) for
faster iteration during protocol-layer development.

## Files

```
   src/tournament-ntk.ts                   federated DO (no test loss, no
                                            byzantine — deferred to next-3)
   src/worker.ts + wrangler.jsonc          binding + migration v10
   scripts/ntk-verifier.py                 Python worker using ntkmirror's
                                            ForwardFineTuner internals
   docs/PHASE_40_NEXT2_NTK_DO.md           this doc
```

## Reproducing

```bash
# 1. wrangler dev (uses local R2 simulation)
cd ~/postnet-cf
npm run dev    # in shell A

# 2. verifier — Python from the ntkmirror venv we set up in Phase 40 scope
~/ntkmirror/.venv/bin/python scripts/ntk-verifier.py \
    --coord http://localhost:8787 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train ~/ntkmirror/examples/math_train.jsonl \
    --artifact public/data/qwen05b-math-gates-k5000.bin \
    --rounds 30 --trials 4 --reset
```

First run downloads Qwen-0.5B (~1 GB) into the HF cache. Subsequent
runs are instant.
