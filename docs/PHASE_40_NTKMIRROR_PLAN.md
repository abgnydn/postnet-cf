# Phase 40 — NTK-Mirror federated controller training (PLAN)

> _Status: scoped + de-risked, not implemented. This doc captures what
> we learned from running NTK-Mirror's demo locally and outlines the
> integration roadmap. The actual phase 40 ship is multiple sessions._

## What NTK-Mirror is (recap)

Sparse signed log-gate controller over a frozen Hugging Face causal LM:

```
   h'[layer, token, channel] = exp(s_{layer, channel}) · h[layer, token, channel]
   |s_{layer, channel}| ≤ max_log_gate  (default 0.05)
```

A controller is `K` gates, each indexed by `(layer, channel)`. Gates are
selected by `|dL/ds_{l,c}|` magnitude (one-time, central). Values `s`
are then trained via teacher-forced loss.

NTK-Mirror does this gradient-based; **postnet-cf will do it via SPSA
tournament**, federated across browser tabs.

## Empirical anchor (today's local run)

Ran `ntkmirror demo --gates 512 --steps 40 --dtype fp32 --device mps`
on Qwen2.5-0.5B-Instruct (math-arithmetic toy task):

| metric         | base   | + controller | Δ      |
|---|---|---|---|
| train NLL      | 1.779  | 1.694        | −0.085 |
| train token acc| 57.7 % | 59.4 %       | +1.7   |
| eval NLL       | 1.762  | 1.677        | −0.085 |
| eval token acc | 61.4 % | 62.9 %       | +1.5   |

Compute on M-series Mac (fp32, mps): gate selection 1.3 s, training
3.7 s. The pipeline works end-to-end. The very-small K=512 + 40-step
budget produces modest but real lift.

**Note on MPS:** the default `bf16` / `auto` dtype on MPS NaN'd at
step 8. `--dtype fp32` is required on Apple Silicon — relevant for any
local gate-selection runs in the future.

## Controller artifact, dissected

For K=512 gates on Qwen2.5-0.5B:

```
   layer_path:        model.layers      (24 decoder layers)
   hidden_size:       896 channels
   gates per layer:   13-32, mean ~21, layer 23 (final) gets 32
                       → spread across ALL layers, no concentration
   channel range:     [18, 877] of [0, 895] → uses the full width
   max_log_gate:      0.05      (mult factor ∈ [exp(-0.05), exp(+0.05)] = [0.95, 1.05])
   trained s values:  ∈ [-0.010, +0.010]   (using ~20 % of budget)
   non-zero fraction: 91.8 % (|s| > 0.001) → almost every gate works
```

**Wire-format math (the headline):**

```
   layer_indices  (K × uint16)  =  K × 2 B
   channel_indices (K × uint16) =  K × 2 B
   raw values     (K × float32) =  K × 4 B
                                =  K × 8 B total

   K =    512  →   4 KB controller (just verified)
   K =  5 000  →  40 KB controller  ← ntkmirror's published default
   K = 10 000  →  80 KB controller
```

A `(layer, channel)` pair list is **static** — selected once,
distributed to all workers as part of the snapshot. Workers only
SPSA-update the `raw[K]` vector. **Per-round wire still 20 bytes**
(seed + scalar_g + delta) — Phase 39's protocol unchanged.

## Critical design choice — K must be ≥ 5 000

Phase 37 empirically established the SPSA-vs-flip crossover:

| P (trainable params) | winner            |
|---|---|
| 2 379                | flip-and-accept   |
| 31 707               | SPSA tournament   |

K=512 controller sits IN flip-and-accept's regime. For Phase 40 to
demonstrate the SPSA tournament's strength on real LLM training,
**K must be ≥ 5 000**.

Bonus: that matches ntkmirror's published default (`--gates 5000`),
so we're shipping the well-tested configuration.

## Per-trial compute budget (feasibility)

The expensive part: each SPSA trial needs **3 forward passes** through
the frozen base model at three different gate configurations
(θ+εu, θ−εu, θ−η·g·u).

| backend                          | one forward pass (Qwen-0.5B, seq~64) | per trial | per round (4 trials) | per 100 rounds |
|---|---|---|---|---|
| M-series Mac, fp32, mps (local)  | 50-200 ms                            | 150-600 ms| 0.6-2.4 s            | 1-4 min        |
| Browser, Transformers.js (wasm)  | 200-1000 ms (estimate)               | 0.6-3 s   | 2.4-12 s             | 4-20 min       |
| Browser, neuropulse WGSL engine  | 100-400 ms (estimate)                | 0.3-1.2 s | 1.2-4.8 s            | 2-8 min        |

Tractable. For the browser path, neuropulse's WGSL engine
(`~/neuropulse/src/engine/inference.ts` + 11 kernels) is the natural
choice — it already runs Phi-3-mini forward in the browser.

## The four pieces of Phase 40

```
   1. central gate-selection script  (Python, one-time per base model)
      ────────────────────────────────────────────────────────────────
      input:  base model name + a representative training corpus
      output: agnews-mini-style binary artifact:
                [uint32 K][uint32 layers][uint32 hidden][uint32 model_id]
                [K × uint16 layer_indices]
                [K × uint16 channel_indices]
                [K × float32 raw_init (all zeros)]
      shipped as public/data/qwen05b-{taskname}-gates-k5000.bin
      ~40 KB

   2. TS port of gate-apply hook                         (in postnet-cf)
      ────────────────────────────────────────────────────────────────
      a hook function injected into the worker's forward pass that does:
        h'[token, channel] *= exp(s_{layer, channel})
      for each (layer, channel) in the controller's index list.
      Needs to hook into whatever inference path the worker uses
      (neuropulse's TS or a Transformers.js wrapper).

   3. new postnet DO                            (src/tournament-ntk.ts)
      ────────────────────────────────────────────────────────────────
      fork of TournamentHeadSpsaAdaptive (Phase 39's sym-AIMD DO):
        - state: theta = raw[K] (the trainable scalars only)
        - constructor: load the gate-selection artifact from ASSETS
        - apply path: identical to Phase 39's SPSA + sym-AIMD on η
        - byzantine defense: identical to Phase 39 (post-apply real_Δ)
      Architecture constants come from the loaded artifact, not
      hard-coded — this DO works with ANY (model, task) gate set.

   4. browser worker                             (public/ntk-worker.js)
      ────────────────────────────────────────────────────────────────
      load Qwen-0.5B once (cached via OPFS like neuropulse does)
      load the gate-selection artifact + current raw values via snapshot
      each round:
        - SPSA trial: pick seed_t, compute θ±εu, run 3 forward passes
          through Qwen-0.5B with the hook injected, get loss values,
          derive scalar_g and claimed_Δ
        - report (seed, scalar_g, delta) to the server
        - reconcile applied_history (replay sym-AIMD η updates)
      this is where neuropulse comes in: reuse its WGSL inference engine
      so a 0.5B-param forward is browser-runnable in ~hundreds of ms.

   stretch — composability artifacts
      ────────────────────────────────────────────────────────────────
      every N rounds, snapshot the current raw[K] as a "memory item":
        public/data/memories/{task-id}-{round}.bin   (~20-40 KB each)
      a separate UI surfaces a "library of skills"; combining them at
      inference time = sum of log-gates (clipped to budget). This is
      where the federated story gets paper-grade interesting.
```

## Open design questions

1. **Which base model?**
   - Qwen2.5-0.5B (~250 MB int4, ~1 GB fp16): smallest viable, downloads
     in seconds, ~150 ms forward on Mac. Browser-friendly.
   - Phi-3-mini-4k-instruct (~2 GB int4, ~8 GB fp16): neuropulse already
     has it running. Heavier per-tab download.
   - Probably **Qwen-0.5B for v1**, Phi-3 as a stretch.

2. **Where do the base weights live?**
   - Option A: each worker downloads from HuggingFace (~1 GB per
     unique browser, cached in OPFS like neuropulse).
   - Option B: we proxy through Cloudflare (paid R2 egress).
   - **Option A**, matching neuropulse's pattern.

3. **Which task for the first demo?**
   - Math arithmetic (matches ntkmirror's demo; tiny dataset).
   - AG News topic classification (reuses our Phase 38 features —
     but those are sentence-embedding features, not raw text; would
     need a different formulation for an LM controller).
   - Style transfer (e.g., "respond in pirate voice") — fits the
     composable-memory pitch better.
   - **Math first**, style transfer as the demo headline.

4. **Browser inference path: Transformers.js or neuropulse WGSL?**
   - Transformers.js: easier to integrate, slower, more memory.
   - neuropulse WGSL: faster, harder to integrate (need to surface
     the hook-injection point), but it's the engine we said we'd use.
   - **Start with Transformers.js**, swap in neuropulse WGSL as
     the perf upgrade once the protocol layer is validated.

## Ship sequence (multi-session)

```
   session  scope                                            ~effort
   ───────────────────────────────────────────────────────────────
   this     scope + dry-run ntkmirror locally  (DONE)        ~1 hour
   ────────────────────────────────────────────────────────────
   next 1   port gate-apply hook to TS; write gate-           ~3 hours
            selection Python script that outputs the
            binary artifact; bake K=5000 gates for one
            (base, task) pair as a static asset
   ────────────────────────────────────────────────────────────
   next 2   write src/tournament-ntk.ts (DO); wire route +    ~3 hours
            migration; verifier using Transformers.js for
            the forward pass; first end-to-end run R=50
   ────────────────────────────────────────────────────────────
   next 3   browser worker (public/ntk-worker.js +            ~half day
            ntk.html); end-to-end demo: open a URL,
            join the swarm, train a math controller
   ────────────────────────────────────────────────────────────
   stretch  composability artifacts: snapshot raw[K] every    +half day
            N rounds as a "skill memory"; UI to list +
            combine memories; demo: train two skills
            independently, compose them, eval combined
   ────────────────────────────────────────────────────────────
   far      swap Transformers.js for neuropulse WGSL engine;  +1 day
            Phi-3-mini base; the real "browser tabs of
            strangers federate-train a Phi-3 skill module"
            headline.
```

Total roadmap: ~3-4 focused sessions for the v1 demo, ~1 more
for the WGSL+Phi-3 upgrade.

## What this session validated

- ntkmirror installs cleanly on Python 3.14 + torch 2.12 in a venv.
  No surprises.
- Qwen2.5-0.5B-Instruct downloads + loads + runs forward in seconds
  on M-series MPS (with `--dtype fp32`).
- The `.pt` controller artifact is exactly the shape we expected:
  three small tensors (layer_indices, channel_indices, raw values).
  Trivially portable to a static-asset binary format.
- Training works end-to-end: gate selection (1.3 s) + 40 training
  steps (3.7 s) → measurable lift on a real benchmark.
- The wire-format math holds: K=5000 gates = 40 KB static controller,
  per-round wire stays at Phase 39's 20 bytes.

**Phase 40 is plausible.** No infrastructure showstoppers found.

## Notes for the next session

- Don't forget `--dtype fp32` for any Mac MPS run, or training NaN's
  immediately. We hit this on the first demo attempt.
- The ntkmirror Python `SignedLogMaskState` class
  (`src/ntkmirror/controller.py`) is the canonical reference for
  the gate-apply math when porting to TS.
- The `compose()` function (`src/ntkmirror/compose.py`) handles the
  additive-in-log-space combination. Port this too when we get to
  the composability stretch.
- neuropulse's `src/engine/inference.ts` exposes a per-layer hook
  point that we can intercept for gate application — verify exactly
  where when starting next-1.
