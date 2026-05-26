# Open questions

Things this implementation does not address, in roughly decreasing order of how soon you'd hit them in a real deployment. Most are open research directions, not just todos.

## Sybil resistance

The byzantine defense quarantines per worker-id. An attacker that rotates IDs every 10 wins evades indefinitely. Mitigations require a stable identifier:

- **DKIM-signed updates** — bridge to the email-based `~/postnet/` transport: a worker's proposals carry a DKIM signature from its sending domain. Cross-session reputation accumulates against the domain, not the in-session worker id.
- **Capability tokens** — single-issue credentials minted by the coord (or by a trusted upstream). Each worker burns one token per submission. Token issue can be rate-limited or proof-of-work-gated.
- **IP / SNI fingerprinting** — coarse and easily defeated by VPN rotation, but raises the cost of bot-net attacks.

Useful in combination: DKIM for identity, capability tokens for rate, IP heuristics for sanity checks.

## Patient-patient attackers

A worker that acts honest for 50 wins, then attacks briefly, then returns to honest can sneak a small number of fraudulent flips through. The sliding window currently caps at 20 verdicts — within that, the attacker can sustain a brief attack burst that doesn't trip the 40% threshold.

Open directions:

- **Tiered windows** (last 20, last 100, last 1000) with independent thresholds. A long-term attacker accumulates fraud in the largest window even if short-window bursts are small.
- **Exponentially-weighted fraud rate** instead of a strict window. Smooth penalty function with no hard "before/after" boundary.
- **Fraud-cost amortization**: when a fraud is detected, retroactively roll back the last N applied flips from that worker (treat the entire recent history as suspect once the worker is flagged).

## True per-worker θ divergence

All current variants assume θ is a single shared parameter vector. Real federated learning often runs per-client local updates (FedAvg = multiple local SGD steps per round). The flip-and-accept protocol could be extended:

- Worker proposes a *trajectory* of K flips instead of a single flip — more aggressive local search but harder to verify cheaply.
- Worker proposes *deltas relative to the worker's local θ* (not the coord's). Coord transports proposals with full local-θ context; aggregates via FedAvg-style weighted average.

The trade-off is bandwidth (trajectories grow with K) and security (verifying a multi-step trajectory's claimed delta is more expensive than a single-step delta).

## Multi-coord federation

A single Coord is the bottleneck. Real distributed training would have N regional coords, each aggregating its local workers, and a root coord aggregating across regional coords. Implementation sketch:

```
workers (region A) → leaf-coord A ─┐
                                   ↓
workers (region B) → leaf-coord B → root-coord ↔ delta-only
                                   ↑
workers (region C) → leaf-coord C ─┘
```

The root coord runs an alarm (`state.setAlarm(t)`) on a slower clock and pulls each leaf's recent `applied_since`, merges them, and pushes the merged θ back. Leaves can train independently between root-syncs.

The merge step is the open algorithmic question: how do you combine N independently-evolved θs into one? FedAvg-style mean is the obvious baseline; SWA or model souping are alternatives worth measuring.

## Realistic model scale

The char-LM is 2 379 params on 340-char text. Far below where bandwidth scaling matters in absolute terms. The protocol is *designed* to scale to BitNet b1.58 2B (P = 1.5B, ternary), but no production-tested end-to-end run exists at that scale.

Concrete blockers:
- **fused-lora's BitNet port** is in progress (per `~/Documents/GitHub/fused-lora/BITNET_PORT.md`); the engine currently runs Phi-3-mini with 10 hand-written WGSL shaders.
- **Worker WebGPU initialization** has not been integrated. The worker's local scorer is still a pure-JS forward pass; for BitNet 2B that's impossibly slow.
- **Bootstrap from R2** at ~282 MB has not been measured under real wall-clock. The sharded bootstrap path exists; needs an end-to-end run to validate the parallel-fetch model.

## Persistent DO state

The current implementation keeps `theta`, `appliedHistory`, `workerStats` in instance variables — all in memory. A Cloudflare DO that gets evicted or restarted loses everything. The `new_sqlite_classes` migration is declared but `state.storage.*` calls are not used.

For long-running training this matters: a several-hour run would lose progress on any platform-level eviction. Fix is straightforward (mirror θ + history to storage on each round advance), at a per-round latency cost worth measuring.

## End-to-end attacker recovery

When the defense detects a fraud, it currently:

- ✅ Quarantines future proposals from that worker
- ❌ Does **not** roll back already-applied fraudulent flips

A determined attacker that gets through the 10-win burn-in window applies several bad flips before getting caught. The model has to recover by training over them. An ideal defense would *undo* the bad applied flips (rewind θ to before they were applied, replay subsequent honest flips). The `appliedHistory[]` array has enough information to do this; the rollback logic doesn't exist yet.

Open question: what does "rewind θ" mean when many other honest flips have been applied since the bad one? Strict rewind loses honest work. Best plausible answer: maintain a "before-flip" snapshot for the last K applied flips so any one of them can be reverted; if older fraud is detected, accept the loss and let training continue.

## What this work doesn't claim

- It is **not a competitive language model**. The 2 379-param char-LM on 340-char text demonstrates the protocol, not the model quality.
- It is **not a peer-reviewed paper**. `docs/PAPER_DRAFT.md` is a working draft; numbers are repeatable but small-sample.
- It is **not production-grade FL**. There is no encryption-of-updates story, no auditable training log, no signed model release path. Those are open architectural questions, not solved problems.

What it *does* claim: a complete reference implementation of a federated flip-and-accept protocol with byzantine defense, bandwidth-optimal asymptotics, and real empirical convergence on a non-trivial task — all running on Cloudflare's free tier and reproducible from a clean clone in five minutes.
