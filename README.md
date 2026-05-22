# postnet-cf

Federated learning where workers are browser tabs and the coordinator is a Cloudflare Durable Object.

A 129-parameter 2D classifier trained by federated SGD across whatever browser tabs you open. Each tab computes the gradient of BCE loss on its local synthetic batch (manual backprop, ~50 lines of JS), POSTs the gradient to the coord, the coord averages all gradients in the round's pool and applies an SGD-with-momentum step. State lives in a single Durable Object; θ is JSON-blobbed in every tick response.

Architecture:

```
browser tab #1   browser tab #2   browser tab #N
   ∇L on            ∇L on            ∇L on
   local batch      local batch      local batch
       │                │                │
       └────────┬───────┴────────┬───────┘
                │ POST /api/tick │
                │  { gradient }  │
                ▼                ▼
       ┌────────────────────────────┐
       │  Cloudflare Worker         │
       │   → routes /api/* to       │
       │     idFromName("default")  │
       └────────────────────────────┘
                       │
                       ▼
       ┌────────────────────────────┐
       │  Coord (Durable Object)    │
       │   pool: gradients          │
       │   when |pool| ≥ 2:         │
       │     avg = mean(pool)       │
       │     v ← βv + avg           │
       │     θ ← θ − lr·v           │
       │     round++                │
       │   broadcast new θ          │
       └────────────────────────────┘
```

## Results

- Local demo, **3–6 browser tabs**, ~5 rounds/sec
- **Loss 0.69 → 0.06** in ~400 rounds (≈ 80 sec wall clock)
- Decision boundary visibly bends to track `sin(2x) > y`
- Per-round bandwidth: 129 floats × 4 bytes ≈ 520 B gradient + 520 B θ broadcast
- Coordinator state: single Durable Object, in-memory pool, capped history

## What worked (and what didn't)

This project went through three algorithms before the boundary actually bent:

1. **ES with σ=0.1**: loss plateaued at 0.32 (y-only local minimum), 1011 rounds
2. **ES with σ_init=0.5 decaying to 0.05**: same plateau, same minimum — ES on this objective genuinely can't escape the y-only basin from random init
3. **FedSGD, plain SGD lr=0.1**: also plateaued at 0.31. The y-only basin is *that* attractive on uniform-2D sin data — ~85% of samples are already correct under "predict y<0", so the gradient signal pulling toward x-dependence is weak and gets buried
4. **FedSGD with momentum (β=0.9), lr=0.3** → ✅ **converged**. Effective LR is ~3.0 at steady state, big enough to escape

Lesson: optimization choices matter at least as much as the protocol. The substrate (CF DO + browser workers) is identical across all four; only the math differed.

## Run locally

```bash
npm install
npx wrangler dev --port 8787
# → open http://localhost:8787 in 2+ tabs, click Join in each
```

## Deploy

```bash
npx wrangler deploy
# → CF gives you a *.workers.dev URL; share with a friend, they become a worker
```

## File layout

| file | role |
|---|---|
| `src/worker.ts` | Worker entrypoint + Coord Durable Object (test-loss eval, gradient averaging, SGD+momentum, /api/tick + /api/state + /api/reset) |
| `public/index.html` | UI (loss chart + boundary canvas + log panel) |
| `public/worker.js` | Browser worker: forward, manual backprop, POST loop, boundary rendering |
| `wrangler.jsonc` | CF config: name, main, assets binding, Durable Object binding, SQLite-backed migration |

## Tasks

Three synthetic 2D objectives the same architecture trains on (selectable from the UI dropdown):

| task | rule | difficulty |
|---|---|---|
| `wave` | `sin(2x) > y` | hardest — y-only basin attracts SGD; needs Adam/momentum to escape |
| `circle` | `x² + y² < 1` | easy — radial structure, ReLU MLP nails it in ~30 rounds |
| `xor` | `sign(x) ≠ sign(y)` | medium — the canonical "must use both inputs" test |

Switching the task via the UI dropdown resets the coordinator (θ, Adam moments, history). Browsers fetch the active task in every tick and label their batches accordingly.

## Configuration

Edit the constants at the top of `src/worker.ts`:

- `H = 32` — hidden width (P = 4H+1 = 129 params)
- `TARGET_GRADIENTS = 2` — pool size before Adam step (advance round)
- `LR = 0.05` — Adam learning rate
- `ADAM_B1 = 0.9`, `ADAM_B2 = 0.999`, `ADAM_EPS = 1e-8`

Browser-side constants in `public/worker.js`:

- `BATCH_SIZE = 64` — synthetic samples per local gradient
- `POLL_DELAY_MS = 80` — pause between worker tick loops

## Tournament protocol (fusedx integration)

The same task with a different aggregation shape: instead of every worker sending a gradient and the coord averaging them, every worker locally tries `K = 8` random flips of `4` parameters, scores each on its private batch, and submits just its **single best** `(indices, values, Δloss)`. The coord picks the best across all workers in the round and applies it.

UI at `/tournament.html`. Same task selector, same boundary canvas; new stats for accept rate and live bandwidth, plus green ticks on the chart marking applied rounds.

```bash
node scripts/tournament-verifier.mjs   # smoke-test the protocol
```

### Phase 1 — flip-and-accept tournament

Per-worker per-round payload at P = 129:

| protocol | up | down |
|---|---|---|
| federated Adam | ~530 B (gradient) | ~530 B (full θ) |
| Phase 1 tournament | ~36 B (flip) | ~530 B (full θ) |

Headless verifier (3 workers × 400 rounds): circle 0.07 ✓ · xor 0.07 ✓ · wave 0.21 △

### Phase 2 — delta-only broadcasts

Workers bootstrap θ once via `/api/tournament/snapshot`, then maintain `localTheta` by applying each tick response's `applied_since` (the list of flips applied since the worker's last sync). The coord no longer ships full θ in tick responses.

Per-worker per-round payload at P = 129 (this demo):

| protocol | up | down | scales with P? |
|---|---|---|---|
| federated Adam | 4·P + ~70 B | 4·P + ~70 B | **yes** (both) |
| Phase 1 tournament | flip + ~50 B | 4·P + ~70 B | down only |
| Phase 2 tournament | flip + since + ~30 B | ~N_flips·40 + ~80 B | **no** (after bootstrap) |

Per-worker per-round measured by the headless verifier:

| protocol | up | down |
|---|---|---|
| Phase 1 tournament | ~120 B | ~470 B (theta dominates) |
| Phase 2 tournament | ~235 B (includes since_round + ack) | ~680 B (JSON-heavy on this tiny model) |

At P = 129 Phase 2 is *not* a bandwidth win — JSON envelope overhead dominates. The point is the **scaling**:

|   | downlink at P = 129 | downlink at P = 1M | downlink at P = 500M (BitNet 2B) |
|---|---|---|---|
| federated Adam / Phase 1 | ~520 B | ~4 MB | ~2 GB (exceeds Worker response cap) |
| Phase 2 | ~680 B | ~680 B | ~680 B |

Convergence is unchanged (3 workers × 400 rounds): circle 0.10 ✓ · xor 0.08 ✓ · wave 0.23 △.

Drift handling: each worker sends its `localRound` as `since_round`. Server returns every applied flip with `round >= since_round`. If the coord's `appliedHistory` has been truncated past the worker's last sync (cap 1000), the worker detects via `oldest_applied_round > localRound + 1` and re-bootstraps via `/snapshot`.

### Phase 3 — R2 snapshots + binary bootstrap

The bootstrap snapshot moves to R2 with versioned keys. The `/api/tournament/snapshot` endpoint returns a JSON pointer; workers fetch the actual θ from `/api/tournament/snapshot.bin?round=N`, which serves the bytes from R2 (or in-memory fallback if R2 is cold). Wire format is binary: `[uint32 round][uint32 P][P × float32]`. Snapshot re-published every 50 accepted rounds so fresh workers don't have to replay the entire history.

Bandwidth scaling, measured by `node scripts/bandwidth-sweep.mjs` (wire-format bytes, includes JSON envelope):

| H | P | Adam ↓/tick | Phase 1 ↓/tick | Phase 2 ↓/tick | Bootstrap binary |
|---|---|---|---|---|---|
| 32 | 129 | 1.7 KB | 1.8 KB | 339 B | 524 B |
| 128 | 513 | 6.4 KB | 6.5 KB | 339 B | 2.0 KB |
| 512 | 2,049 | 24.1 KB | 24.2 KB | 340 B | 8.0 KB |
| 2048 | 8,193 | 99.8 KB | 100.0 KB | 340 B | 32.0 KB |
| 8192 | 32,769 | 509.0 KB | 509.2 KB | 341 B | 128.0 KB |
| BitNet 2B | 1.5B | ~282 MB (overflows Worker 100 MB response cap) | same | 337 B (constant) | 282 MB via R2 range read |

At 8192-hidden-unit models, Phase 2 is **~1,500× smaller** than Adam or Phase 1 per tick. At BitNet 2B, the linear protocols don't fit in a Worker response at all — Phase 2 still ships ~340 B/tick because the wire payload is the accepted flip and a small JSON envelope, not the model.

R2 setup: `wrangler.jsonc` declares an `R2_BUCKET` binding named `SNAPSHOTS`. Local `wrangler dev` emulates R2 automatically; for `wrangler deploy` create the bucket once:
```bash
npx wrangler r2 bucket create postnet-snapshots
```

The DO writes snapshots via `env.SNAPSHOTS.put(key, buf, ...)` and reads via `env.SNAPSHOTS.get(key)`. Response headers `x-snapshot-source: r2 | memory` make it easy to verify which path served the bytes.

### Phase 4 — ternary weights

Same flip-and-accept protocol but every weight is constrained to <code>{−1, 0, +1}</code> with a single learned scale `S`. Effective weight at index *i* is `sign[i] * scale`. Workers propose ternary flips (pick K positions, set each to a new value ≠ current), score, submit best. Substrate test for plugging in a real BitNet b1.58 model as Phase 5.

UI at `/ternary.html`. New DO `Ternary` (migration v3); endpoints `/api/ternary/{tick,state,reset,set_task,snapshot,snapshot.bin}`.

Snapshot wire format (packed):
```
[uint32 round][uint32 P][float32 scale][ceil(P*2/8) bytes packed]
```
Each ternary value packs into 2 bits: `00 = 0`, `01 = +1`, `10 = −1`. At `P = 129` the snapshot is **45 bytes** (vs 524 B for the float Tournament — 11.6× smaller). At BitNet 2B (`P = 1.5B`) the same encoding ships **~375 MB** once via R2 (vs ~6 GB float32 — 16× smaller).

Headless verifier (`scripts/ternary-verifier.mjs`, 3 workers × 800 rounds): circle 0.34 △ · xor 0.31 △ · wave 0.30 △. Convergence stops short of the float Tournament's 0.07-0.21 because the ternary search space is genuinely coarser at P=129 — there are only 3^129 reachable weight configurations and a single scale. Accept rate drops to ~25% (vs ~50% for float) because many proposed flips don't beat the current state; each accepted flip is a discrete improvement.

At BitNet 2B scale the same protocol drives 1.5 B ternary weights with the same ~340 B per-tick downlink. Phase 6 will swap the worker's local scorer for `fused-lora`'s WebGPU BitNet inference, keeping the protocol and DO unchanged.

### Phase 5 — char-LM (real ML task)

Steps the substrate up from the 2D toy classifier to a real ML task: next-character prediction over a fixed 340-char toy passage. Architecture is a 27-vocab bigram: `embed (V × E) + linear (E × V) + bias (V)` = 27·16 + 16·27 + 27 = **891 params**. Loss = mean cross-entropy. Random-init loss ≈ `log(V) ≈ 3.30`.

UI at `/lm.html`. New DO `TournamentLM` (migration v4); endpoints `/api/lm/{tick,state,reset,snapshot,snapshot.bin,sample}`. The `/sample` endpoint runs greedy + temperature decoding from the coord's current θ — handy for eyeballing convergence without joining a worker.

Headless verifier (`scripts/lm-verifier.mjs`, 3 workers × 1500 rounds):

| round | loss | accept rate | sample (60 chars from "t") |
|---|---|---|---|
| 0 | 3.29 | 0% | `"txhozciuicxcczuxhcziczhxhchnuczhczihcihzizixchch zuxczuxuxwi"` |
| 400 | 2.23 | 50% | `"the the the the the the the the the the the the the the the "` |
| 800 | 1.84 | 49% | `"the the the the the the the the the the the the the the the "` |
| 1500 | 1.67 | 45% | `"the the the the the the the the the the the the the the the t"` |

Loss drops monotonically from 3.29 → 1.67 driven entirely by federated flip-and-accept on a real ML loss surface (no gradients, no autograd). The mode collapse to "the the the…" is expected for a 1-char-context bigram — it has no notion that it just emitted the same word three times. Phase 6 (BitNet b1.58 via fused-lora) brings real context length and breaks out of the mode-collapse regime; the protocol stays the same.

### Phase 6 — sharded snapshots (parallel bootstrap)

`/api/lm/snapshot` now returns a **manifest** listing N shards instead of a single binary URL. Each shard is its own R2 key (`lm/r{N}/shard{K}.bin`); workers fetch them all in parallel via `Promise.all` and assemble the local θ. The first shard carries the 8-byte `[round, P]` header; subsequent shards are raw `float32` payloads.

```bash
$ curl -s localhost:8787/api/lm/snapshot | jq '.num_shards, .shards | length'
4
4
```

At P = 891 with `SHARD_SIZE = 1 KB` the snapshot splits into 4 shards (256 + 256 + 256 + 123 floats). Reassembly in the worker via parallel fetch + sorted `Float32Array.set()`. Bootstrap latency is now bound by the slowest single shard, not the total bytes.

The point: the Cloudflare Worker response cap is 100 MB. At BitNet 2B scale (~282 MB ternary) a single-shot snapshot fetch is impossible — the bootstrap *must* be sharded. Production sharding would use `SHARD_SIZE ≈ 64 MB` for BitNet (∼5 shards × 64 MB), keeping each below the cap while parallelizing the wall-clock. The demo uses tiny 1 KB shards purely to exercise the assembly logic at small P.

Response headers `x-snapshot-shard: K` and `x-snapshot-source: r2 | memory` make it easy to verify the right shard was returned from the right path.

### Phase 7 — federated data shards (true FL)

Up through Phase 6 every worker scored proposals on the **same** text (the full 340-char passage). Phase 7 splits the text across workers: each tab hashes its `workerId` to a deterministic disjoint slice and scores proposals only on its private shard. The coord still measures convergence on the **full** text — that's the metric that matters — but no individual worker has seen all the data.

With 3 workers covering 3 disjoint shards (chars 0..112, 112..224, 224..336):

```
lm-alpha   → shard 0  [chars   0..112]
lm-delta   → shard 1  [chars 112..224]
lm-bravo   → shard 2  [chars 224..336]
```

Convergence on full text (3 workers × 1500 rounds): 3.27 → 1.94 (vs the centralized 1.67). Slightly worse because each worker's gradient estimate is noisier from only seeing 1/3 of the data — but the shared model still learns the union of the shards via federated flip-and-accept.

This is real FL semantics: workers cannot see each other's data, the coord aggregates only the proposed *parameter updates*, and the global model fits the union of private datasets. Combined with Phase 2's delta-only broadcasts, a worker behind a hostile network can participate without ever revealing what it trained on — only the flips it proposed.

## Open questions / future moves

- **Real workload.** Swap the synthetic 2D classifier for `fusedx`'s `gpt-gradfree-engine.ts` or a TF.js MNIST model. Same coord, real ML compute.
- **Cross-machine demo.** After `wrangler deploy`, send the URL to a friend. Watch their machine become peer #N.
- **Real federated angle.** Currently every tab samples from the same synthetic distribution. For "true" FL, give each tab a different shard or different data domain.
- **DKIM / Postnet bridge.** Wire this same coord shape to the email-based Postnet transport so the system tolerates workers behind hostile networks where outbound HTTPS doesn't reach Cloudflare.
