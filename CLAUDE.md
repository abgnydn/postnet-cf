# ~/postnet-cf — CLAUDE.md

postnet-cf: federated tournament protocol on Cloudflare Workers. 34 phases shipped + deployed 2026-05-26; phases 35–37 (WebGPU substrate, SPSA fusion, scaling crossover) added 2026-05-26/27 — see "Resume here" for state.

## 🎯 Resume here (on "continue")

_Updated: 2026-05-27 (Phases 35–37 staged but uncommitted)_

**Where we are.** The DeComFL fusion landed: SPSA tournament now sits alongside flip-and-accept on the same protocol substrate. Phase 37 empirically validated the scaling crossover — at P=2 379 flip-and-accept wins; at P=31 707 SPSA already wins (1.86 vs 2.43 loss at R=1000). Wire format, byzantine defense, browser-runnable workers all unchanged. See `docs/PHASE_37_SCALING.md`.

**Files staged but uncommitted (as of 2026-05-27):**

- `src/tournament-spsa-lm.ts` + `src/tournament-spsa-lm-big.ts` — SPSA DO (P=2 379 and P=31 707)
- `src/tournament-lm-big.ts` — flip-and-accept at the big scale (control)
- `src/worker.ts` + `wrangler.jsonc` — bindings + migrations for the four new DOs
- `public/lm-webgpu-scorer.js` + `public/lm-parity.html` — WebGPU substrate (parity ✓ but 0.55× speed on tiny model; substrate ready for bigger model)
- `public/spsa-lm-worker.js` + `public/spsa-lm.html` — browser SPSA worker + demo page
- `scripts/spsa-verifier.mjs` + `scripts/lm-big-verifier.mjs` + `scripts/spsa-big-verifier.mjs`
- `docs/PHASE_37_SCALING.md` — empirical writeup
- `README.md` — phase table extended (35/36/37)

**Highest-impact moves remaining (in order):**

1. **Phase 38 — Phi-3 + frozen-backbone head via SPSA.** Wire neuropulse's existing WGSL Phi-3 engine into a new postnet DO as a frozen feature extractor. Train a small task-head (sentiment / classifier, ~6K–30K params) federated via the SPSA tournament. The "real LLM federated training in browser tabs" headline. ~1 week. Bridge file: `~/neuropulse/src/engine/inference.ts` + the 11 WGSL kernels under `~/neuropulse/src/engine/shaders/`. The SPSA forward-only pattern fits this perfectly (no backprop needed on Phi-3).

2. **Phase 37b — multi-seed empirical sweep + wider P range.** Phase 37's numbers are single-sample. For paper-grade evidence: ≥5 seeds per (protocol × P) cell, plus P sweep {5K, 10K, 20K, 50K, 100K}. Maps the crossover precisely. ~1 day, but burns wrangler dev for hours; do via deployed worker.

3. **SPSA + Adam-momentum on the scalar.** Track an EMA of past `scalar_g` per worker, use it as the actual step direction. MeZO-SVRG (arXiv:2404.08080) is the reference. Should halve the rounds needed at the same P. Wire format unchanged (workers still send a single scalar). ~half day.

4. **CYBER-0 stacking** (arXiv:2406.14362). Add trimmed-mean as a pre-filter before SPSA tournament selection. Postnet's post-apply defense becomes the asynchronous reputation layer; CYBER-0's filter handles synchronous batches. Lets us benchmark against published numbers. ~half day.

5. **Persistent DO state** (deferred from prior Resume blocks). Currently all six tournament DOs are in-memory. Declare `state.storage` schema (theta, appliedHistory, workerStats), restore on construct, write on advance. ~30 min per DO.

**The "obvious next move" if you say "go":** Phase 38. Phase 37 gave the empirical anchor for the scaling claim; Phase 38 builds the demo that turns the claim into a thing people can click.

**Quick orientation pointers:**

- `README.md` — landing page + 37-row phase table
- `docs/PHASE_37_SCALING.md` — scaling crossover empirical (this is the "why SPSA matters" doc)
- `docs/PROTOCOL.md` — wire-format spec (v0.5; needs bump to v0.6 to spec SPSA proposal/applied shapes)
- `docs/EMPIRICAL_STUDY.md` — multi-seed numbers (vanilla / sharded / byzantine, attacker-count sweep)
- `docs/PAPER_DRAFT.md` — 8-section arXiv-style writeup (predates Phase 36; needs an SPSA section before submission)
- `docs/OPEN_QUESTIONS.md` — what's not addressed yet (incl. the DKIM / Python / long-async extensions that are out of scope without the email v1 bridge)
- `scripts/empirical-study.mjs` — main driver for flip-and-accept comparisons
- `scripts/spsa-verifier.mjs` / `scripts/spsa-big-verifier.mjs` — SPSA verifiers
- `~/neuropulse/` — Phi-3 WebGPU engine (live demo of forward pass); the integration target for Phase 38

**Verify-by-reading rules:** when reviewing changes to `src/*.ts`, prefer reading over running smoke tests — `wrangler dev` is unstable under multi-verifier load (documented in Phase 19; reconfirmed Phase 37 ECONNRESET at R~1050). Production-deployed Workers handle the load fine.
