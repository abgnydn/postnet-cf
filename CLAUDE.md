# ~/postnet-cf — CLAUDE.md

postnet-cf: federated flip-and-accept protocol on Cloudflare Workers. 29 phases shipped 2026-05-22, deployed 2026-05-26. See README.md and docs/README.md for orientation.

## 🎯 Resume here (on "continue")

_Updated: 2026-05-26 (Phase 29 shipped; substrate complete + deployed)_

**Goal:** This project is feature-complete as a research artifact. The substrate is uniformly resilient (all 4 DOs have byzantine defense, all 3 tournament DOs have WS push + R2 sharded snapshots), the public deploy is live, and there's a paper draft. The next concrete forward moves are listed in `docs/OPEN_QUESTIONS.md`. Pick one based on priority.

**Highest-impact moves remaining:**

1. **WebGPU worker scorer** — replace the JS forward pass in `public/lm-worker.js` with a WGSL compute shader, then plug fused-lora's Phi-3-mini engine. Multi-session lift; the bandwidth substrate is ready. Concrete entrypoint: copy `~/Documents/GitHub/fused-lora/src/zero-tvm/engine-core.ts` and the 10 WGSL shaders into `public/`, write a minimal scorer interface.

2. **Persistent DO state** — currently in-memory; declare a `state.storage` schema (theta, appliedHistory, workerStats), restore in constructor, write on advance. ~30 min refactor per DO.

3. **Multi-coord federation** — N leaf coords + 1 root with DO alarms. Sketch in `docs/OPEN_QUESTIONS.md`. Real new work.

4. **Sybil resistance via DKIM** — bridge to `~/postnet/` email transport, verify worker proposals via DKIM signature. Connects the two postnet implementations.

**Steps for option 1 (WebGPU scorer):**

1. `cd ~/postnet-cf`
2. Read `docs/PROTOCOL.md` to refresh the scorer-side contract (worker just needs to compute `delta` for a proposed flip).
3. Create `public/lm-webgpu-scorer.js` with the same forward/loss interface as the JS version, but powered by WGSL + WebGPU.
4. Feature-detect WebGPU on join; use WebGPU scorer if available, fall back to JS.
5. Run `scripts/lm-verifier.mjs` against the production URL to confirm parity.

**Acceptance for option 1:** char-LM converges at parity with the JS path, with measurably faster `textLoss()` throughput on a hardware that supports WebGPU.

**Quick orientation pointers:**

- `README.md` — landing page + 29-row phase table
- `docs/PROTOCOL.md` — wire-format spec (v0.5)
- `docs/EMPIRICAL_STUDY.md` — multi-seed numbers (vanilla / sharded / byzantine, attacker-count sweep)
- `docs/PAPER_DRAFT.md` — 8-section arXiv-style writeup
- `docs/OPEN_QUESTIONS.md` — what's not addressed yet
- `scripts/empirical-study.mjs` — main driver for new comparisons (MODE=variants|attackers|smart)
- `scripts/bandwidth-sweep.mjs` — static analysis (no live coord needed)

**Verify-by-reading rules:** when reviewing changes to `src/*.ts`, prefer reading over running smoke tests — `wrangler dev` is unstable under multi-verifier load (Phase 19 documented this; production-deployed Workers handle the load fine).
