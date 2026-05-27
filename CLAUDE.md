# ~/postnet-cf — CLAUDE.md

postnet-cf: federated tournament protocol on Cloudflare Workers. 34 phases shipped + deployed 2026-05-26; phases 35–37 (WebGPU substrate, SPSA fusion, scaling crossover) added 2026-05-26/27 — see "Resume here" for state.

## 🎯 Resume here (on "continue")

_Updated: 2026-05-27 (Phases 35–38 staged but uncommitted; Phase 38 capped at R=100 due to wrangler dev instability)_

**Where we are.** Phase 38 shipped: federated training of a 2-layer MLP head on real MiniLM-L6-v2 features (AG News, 4-class topic classification, P=49 796). Both protocols start from byte-identical θ; at R=100, SPSA descended ~3.5× more in loss (1.40→1.30 vs flip's 1.40→1.37) AND moved test accuracy from 32% to 40% where flip stayed flat at 32%. Validates Phase 37's scaling claim on a real downstream task with real LLM features. See `docs/PHASE_38_HEAD.md`.

**Files staged but uncommitted (as of 2026-05-27):**

- `src/tournament-spsa-lm.ts` + `src/tournament-spsa-lm-big.ts` — SPSA DO (P=2 379 and P=31 707) [Phase 36-37]
- `src/tournament-lm-big.ts` — flip-and-accept at the big scale (control) [Phase 37]
- `src/tournament-head-flip.ts` + `src/tournament-head-spsa.ts` + `src/head-model.ts` — head-classifier DOs + shared model [Phase 38]
- `src/worker.ts` + `wrangler.jsonc` — bindings + migrations (now 6 tournament DOs total)
- `public/lm-webgpu-scorer.js` + `public/lm-parity.html` — WebGPU substrate (Phase 35; parity ✓, 0.55× speed on tiny model)
- `public/spsa-lm-worker.js` + `public/spsa-lm.html` — browser SPSA worker + demo page
- `public/data/agnews-mini.bin` — 100-example MiniLM features (~154 KB) [Phase 38]
- `scripts/extract-agnews-features.mjs` — one-shot feature extraction via @huggingface/transformers
- `scripts/spsa-verifier.mjs` + `scripts/{lm,spsa}-big-verifier.mjs` + `scripts/head-{flip,spsa}-verifier.mjs`
- `docs/PHASE_37_SCALING.md` + `docs/PHASE_38_HEAD.md` — empirical writeups
- `README.md` — phase table extended (35/36/37/38)
- `package.json` + `package-lock.json` — added `@huggingface/transformers` devDep

**Highest-impact moves remaining (in priority order; the 2026 paper scan rewrote this list):**

1. **Phase 39 — MEAZO single-scalar adaptive η (arXiv:2605.03869, May 2026).** Claim: in high-D ZO, coordinate-wise adaptive statistics (Adam-style per-param) give NO convergence advantage over a single global scalar step-size. Falsify or confirm on the head-classifier from Phase 38. If MEAZO is right: ship a 5-line adaptive-η tracker as Phase 39. If wrong: fall back to HiSo per-param preconditioner (arXiv:2506.02370). ~half day. The cheapest known fix for Phase 38's slow per-round descent.

2. **Phase 40 — Learnable aggregation weights (arXiv:2511.03529, ICLR 2026).** Upgrades the postnet tournament from argmax-over-K to a jointly-learned linear combination of the K proposals. Pairs with Phase 39 regardless of which scalar-η scheme wins. ~1 day.

3. **Phase 38b — multi-seed sweep + multi-P sweep on head-classifier.** Phase 38's R=100 numbers are single-sample on a 25-example test set. For paper-grade evidence: ≥5 seeds per (protocol × P) cell, P sweep {25K, 50K, 100K, 200K} via H bumps. Run against a deployed Worker (wrangler dev R>100 is unstable). ~1 day.

4. **Phase 41 — VerifBFL zk-SNARK (arXiv:2501.04319).** Replaces post-apply byzantine check with a cryptographic guarantee that the worker computed `scalar_g` honestly on its committed data shard. 0.6 s on-chain verify, 81 s proof gen per worker. The "splashy security" upgrade. Positions postnet against Gensyn/Prime Intellect's "verifiable training" pitch but for a SINGLE SCALAR rather than a full gradient (orders of magnitude cheaper to prove). ~1 week.

5. **Persistent DO state** (deferred from prior Resume blocks). Currently all six tournament DOs are in-memory. Declare `state.storage` schema (theta, appliedHistory, workerStats), restore on construct, write on advance. ~30 min per DO.

**Strategic context from the May 2026 paper scan:** INTELLECT-3 (Prime Intellect, Nov 2025) went centralized — trained 106B MoE on 512 H200s in one cluster, abandoning INTELLECT-1/2's decentralized story. The "anyone-can-join" frontier in mid-2026 is effectively Nous DisTrO + Pluralis + postnet-cf. The competition thinned; postnet's browser-tab niche is MORE differentiated than 6 months ago, not less.

**The "obvious next move" if you say "go":** Phase 39 (MEAZO falsification). Cheapest possible empirical win; collapses the next two phases into one if their claim holds.

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
