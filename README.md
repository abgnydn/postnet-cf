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

## Open questions / future moves

- **Real workload.** Swap the synthetic 2D classifier for `fusedx`'s `gpt-gradfree-engine.ts` or a TF.js MNIST model. Same coord, real ML compute.
- **Cross-machine demo.** After `wrangler deploy`, send the URL to a friend. Watch their machine become peer #N.
- **Real federated angle.** Currently every tab samples from the same synthetic distribution. For "true" FL, give each tab a different shard or different data domain.
- **DKIM / Postnet bridge.** Wire this same coord shape to the email-based Postnet transport so the system tolerates workers behind hostile networks where outbound HTTPS doesn't reach Cloudflare.
- **LR decay.** Current loss curve oscillates due to momentum + constant LR. A `lr / sqrt(1 + round/100)` schedule would smooth it.
- **Adam.** Would handle the plateau-escape automatically without manual momentum tuning. ~80 lines extra in the coord.
