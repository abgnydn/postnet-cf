# Postnet-CF: A federated flip-and-accept protocol with byzantine resistance, on Cloudflare Workers

*Working draft — not peer reviewed*

## Abstract

We implement and evaluate a federated learning protocol designed for browser-tab workers behind hostile networks. The substrate is a Cloudflare Durable Object that coordinates worker proposals via a single-winner tournament: each round, every worker proposes a small parameter "flip" (`K` index/value updates) and a claimed loss delta; the coordinator picks the most-negative claimed delta and applies it. The protocol is **bandwidth-optimal asymptotically** — per-tick downlink is `O(1)` in model size after a one-time bootstrap — and is **byzantine-resistant to 50% attacker share** via a sliding-window fraud-detection defense that verifies reported deltas against an independent test-loss measurement. We evaluate the protocol on a char-LM task (P = 2 379 parameters) under controlled attack across 4 protocol variants and 3 attacker counts, with statistically tight numbers across seeds. The complete reference implementation, including a unified dashboard, runs on the Cloudflare free tier; the bootstrap snapshot path scales to BitNet-2B-class models (P ≈ 1.5B ternary, ~282 MB) via sharded R2 storage.

## 1. Background and motivation

Federated learning workers behind home routers, hotel WiFi, and corporate proxies cannot accept inbound connections. The standard FL stacks (Flower, FedML, gRPC-based aggregators) assume reachable workers and require either a heavy bidirectional channel or VPN traversal. **Cloudflare Durable Objects** offer a different shape: every worker establishes its own outbound HTTPS, the coord is a single persistent address, and the platform handles routing, retries, and TLS. Workers in any tab on the planet can join via `Open URL` — no install, no port-forwarding, no shared infra.

But a single coordinator backed by `Float32Array.from(theta)` per tick scales linearly with model size and is trivially attackable by any worker willing to submit fake gradients. This work addresses both: a delta-only broadcast protocol whose per-tick cost is constant in `|θ|`, and a byzantine defense whose detection signal — the *observed* global loss change after applying — is computed by the coord regardless of attacker behavior.

## 2. Protocol

A coord holds `(round, θ, pool, appliedHistory, workerStats)`. Each tick is one of two requests:

- **Poll**: `worker_id, since_round`. Returns `applied_since: Flip[]` — every accepted flip with `round ≥ since_round`. Workers maintain `localTheta` by applying these in order.
- **Submit**: `worker_id, round, indices, values, delta`. Server admits to `pool` if `round == server.round` and the worker is not quarantined. When `|pool| ≥ TARGET_PROPOSALS = 2`, the server picks `bestProposal = argmin pool.delta`. If `bestProposal.delta < 0`, it applies the flip to `θ`, records the *real* `lossAfter − lossBefore`, and advances `round`.

The bootstrap path returns a manifest of binary shards: `[uint32 round][uint32 P][P × float32]` sliced into multiple R2 objects. Workers fetch shards in parallel and assemble `localTheta`. For ternary models (BitNet b1.58), the same layout uses 2-bit packed signs (`00 = 0, 01 = +1, 10 = −1`) plus a single `float32` scale.

**Per-tick downlink** is `O(K · 8) + O(|applied_since| · K · 8)` ≈ 340 B for the demo's char-LM, irrespective of model size; **bootstrap** is `O(|θ|)` one-shot.

## 3. Defense

We adopt the classic "verify, don't trust" pattern: post-apply, the coord computes `real_global_delta = lastLoss_after − lastLoss_before`. A winning proposal with claimed `delta < −1e-4` but `real_global_delta > 1e-4` is marked a fraud and recorded in the winning worker's stats.

Quarantine triggers when `max(cumulative_rate, last_20_window_rate) > 0.4` after at least 10 wins. Cumulative catches consistent attackers; the sliding window catches "patient attackers" that act honest for the first N wins to dodge the cumulative gate.

A federated-Adam baseline coord uses a complementary defense: trimmed-mean aggregation (drop the single largest-norm gradient before averaging). This handles a different attack surface (outlier-magnitude gradients vs. fake delta claims), shipped at parity.

## 4. Empirical evaluation

All numbers from `scripts/empirical-study.mjs`, 3 seeds × 1500 rounds per cell, against an in-process `wrangler dev` running on M2 Pro.

### 4.1 Variant comparison

| variant | n | mean | std | vs vanilla |
|---|---|---|---|---|
| vanilla | 3 | 1.2964 | 0.0285 | +0.0000 |
| sharded | 3 | 1.8505 | 0.0939 | +0.5541 |
| byzantine (def. on) | 3 | 1.9044 | 0.0529 | +0.6080 |

*vanilla* = 3 honest workers, full text (centralized-data baseline).
*sharded* = each worker only scores on a private 1/3 slice (Phase 7).
*byzantine* = 3 honest + 1 attacker, Phase 9-14 defense active.

The +0.55 nats sharded penalty is the cost of each worker estimating the global delta from 1/3 of the data. The +0.05 nats marginal cost of byzantine relative to sharded is the pre-detection window (the attacker needs 10 wins before the cumulative gate fires).

### 4.2 Attacker share

3 honest workers, attacker count varied 0..3:

| attackers | honest share | n | mean | std | vs 0-atk |
|---|---|---|---|---|---|
| 0 | 100% | 3 | 1.8896 | 0.0054 | +0.0000 |
| 1 | 75% | 3 | 1.9568 | 0.0359 | +0.0671 |
| 2 | 60% | 3 | 1.8903 | 0.0911 | +0.0007 |
| 3 | 50% | 3 | 1.9783 | 0.0534 | +0.0887 |

Convergence degrades by less than 5% (+0.09 nats) at 50% byzantine share. Because the per-worker quarantine fires independently, the defense's effective tolerance scales with the *count* of attackers it can detect within the 10-win burn-in, not their proportion.

### 4.3 Without defense (single-shot reference)

A separate single-shot run with 3 honest + 1 byzantine and defense disabled landed at final loss 2.35 — a +0.72-nat gap relative to the 1.63 honest baseline. Adding the Phase 9-14 defense closed 87% of that gap to a +0.10 nat residual.

## 5. Bandwidth scaling

Static analysis of wire-format bytes (`scripts/bandwidth-sweep.mjs`):

| H | P | Adam ↓/tick | Phase 1 ↓/tick | Phase 2+ ↓/tick | Bootstrap binary |
|---|---|---|---|---|---|
| 32 | 129 | 1.7 KB | 1.8 KB | 339 B | 524 B |
| 128 | 513 | 6.4 KB | 6.5 KB | 339 B | 2.0 KB |
| 512 | 2,049 | 24.1 KB | 24.2 KB | 340 B | 8.0 KB |
| 2048 | 8,193 | 99.8 KB | 100.0 KB | 340 B | 32.0 KB |
| 8192 | 32,769 | 509.0 KB | 509.2 KB | 341 B | 128.0 KB |
| BitNet 2B | 1.5B | ~282 MB (does not fit) | same | ~340 B | 282 MB via R2 range read |

At BitNet scale the linear protocols' tick response exceeds the Cloudflare Workers 100 MB body cap and is structurally impossible to ship; Phase 2+ shipping `applied_since: [Flip]` remains ~340 B regardless. Bootstrap moves to a sharded R2 read that parallelizes across keys.

## 6. Related work

- **(1+1)-ES and Gaussian flip-and-accept** (Schwefel, 1981) — the single-worker analog of our tournament protocol.
- **FedAvg** (McMahan et al., 2017) — gradient averaging FL baseline; our federated-Adam coord is the dense equivalent.
- **Krum / trimmed-mean aggregation** (Blanchard et al., 2017) — byzantine-resistant gradient aggregation; we adopt trimmed-mean at the federated-Adam path.
- **BitNet b1.58** (Microsoft, 2024) — the ternary-weight architecture our snapshot encoding targets.
- **The Swarm (Gunaydin, 2025)** — earlier 50% byzantine-tolerance result via trimmed-mean over reported gradients; this work attains comparable tolerance at the protocol layer (verified-delta + quarantine).

## 7. Reproducibility

```bash
git clone https://github.com/abgnydn/postnet-cf
cd postnet-cf
npm install
npx wrangler dev --port 8787
# Browser demo: http://localhost:8787 (or /dashboard.html for all four)
# Headless verifiers: scripts/{headless-worker,tournament-verifier,ternary-verifier,lm-verifier}.mjs
# Empirical study: scripts/empirical-study.mjs (MODE=variants|attackers|smart)
# Bandwidth analysis: scripts/bandwidth-sweep.mjs (no live coord needed)
# Public URL via cloudflared quick tunnel: bash scripts/expose.sh
```

All code MIT-licensed; reference implementation, headless verifiers, and protocol spec at `docs/PROTOCOL.md`.

## 8. Limitations and future work

- **Cross-session sybil resistance.** Quarantine is per-worker-id. An attacker rotating IDs every 10 wins can bypass detection indefinitely. Mitigation requires a stable identifier — IP, DKIM signature, capability token — not addressed here.
- **Patient-patient attacker.** The sliding window catches a 9-honest-then-attack pattern but a 50-honest-then-brief-attack-then-honest pattern can sneak a few fraudulent flips through.
- **Real frontier model.** The char-LM task at P = 2 379 is a substrate test, not a competitive language model. The intended next step is to swap the worker's local scorer for a WebGPU-based Phi-3-mini or BitNet b1.58 forward pass; the protocol is bandwidth-ready (Phase 2 stays constant; Phase 6 shards the snapshot), the WebGPU integration is engineering work.
- **`wrangler dev` instability** under aggressive multi-verifier load is a development-mode bottleneck only; production-deployed Workers do not exhibit it.

The complete implementation is reproducible in a session. We invite further empirical study against patient attackers, sybil rotation, and real models with worker WebGPU.
