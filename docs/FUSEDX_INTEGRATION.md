# Wiring fusedx as a postnet-cf worker

Plan to replace the synthetic 2D classifier with `fusedx`'s real ML workload while keeping the same Cloudflare Durable Object coord substrate.

## What we have

- `~/postnet-cf` — CF Worker + Coord DO + browser worker. Currently trains a 129-param 2D classifier with federated Adam. Bandwidth: ~520 B/round/worker. Substrate validated.
- `~/Documents/GitHub/fusedx` — Next.js app with `gpt-gradfree-engine.ts` (~535 lines): ternary GPT, WebGPU forward pass, **flip-and-accept** evolutionary search.
- `~/Documents/GitHub/fused-lora` — BitNet b1.58 2B inference + LoRA training in browser, 86 ms/step on M-series, `.flora` adapter format (~4 MB, FLASC-pruned to 200-400 KB).

## The mismatch we have to handle

fusedx's algorithm is **not Gaussian ES**, it's **flip-and-accept**:
- Each step: perturb K ternary weights (`{-scale, 0, +scale}`) to random new values
- Run forward + loss on a batch
- If new loss < current loss: keep the flip
- Else: revert

This doesn't aggregate by averaging gradients. It's a tournament: many workers try different flips, the best flip wins.

Two integration shapes:

### Shape A — Tournament coord (matches fusedx semantics)

Each round:
1. Coord broadcasts current θ + a batch seed (for reproducible loss comparison)
2. Each worker downloads θ, generates K random flip proposals, scores each (forward loss)
3. Worker reports its **single best** `(flip_indices[], flip_new_values[], delta_loss)`
4. Coord picks the proposal with the largest negative `delta_loss` across all workers
5. Coord applies that flip to θ, broadcasts new θ
6. Repeat

Wire format per worker → coord:
```json
{
  "worker_id": "w-abc",
  "round": 47,
  "flip_indices": [12384, 19022, 87, ...],   // K ternary indices
  "flip_new_values": [0.07, -0.07, 0.0, ...], // K target values
  "delta_loss": -0.0042                        // observed improvement
}
```

Per-round bandwidth: ~K × (4 + 4) = K × 8 bytes per worker. For K=200 (fusedx default), ~1.6 KB/worker. **Independent of model size.** Even a 2B BitNet model's flip proposal is 1.6 KB.

θ broadcast is the only fat message. For BitNet 2B at 1.58 bits/param, ~400 MB raw. **R2 of the Postnet bandwidth design** (delta-only after bootstrap) is the answer there — coord broadcasts only the accepted flip per round (8 bytes) and workers reconstruct θ locally.

### Shape B — Gradient coord (force-fit fusedx into our existing protocol)

Keep our current Adam/SGD coord; modify fusedx to compute a *gradient estimate* (e.g., via finite differences over a flip) instead of accept/reject. Sketchy — defeats fusedx's design.

**Pick Shape A.**

## Concrete change list

### Coord side (~150 lines added/changed in `src/worker.ts`)

New endpoint or new mode for the existing tick endpoint:
- POST body: `{ worker_id, round, flip_indices: number[], flip_new_values: number[], delta_loss: number }`
- Coord stores best proposal per round in `this.bestProposal`
- After N reports (or timeout), coord applies `bestProposal` to `this.theta`
- Optional: keep a small Adam-like momentum on top of which flips get accepted

Replace the test-loss eval. We don't run a held-out test set server-side anymore (model is too big to forward in a Worker). Instead the coord just tracks `delta_loss` reports from workers as the convergence signal.

### Browser worker side

Three options for the worker:

1. **Vendor fusedx's gradfree engine** into `public/`. Copy `gpt-gradfree-engine.ts` + its WGSL pipelines + the data loader. Compile with esbuild/Vite to a bundle. Big lift (~1 day) — fusedx wasn't designed to be embedded.

2. **Iframe fusedx**, postMessage the coord protocol. Run fusedx's Next.js app at its own URL (deployed separately), embed it via `<iframe>` in postnet-cf, the parent page wraps `fetch` calls to route through `/api/tick`. Messy.

3. **Modify fusedx itself.** Add a `POSTNET_COORD_URL` env to fusedx; when set, the gradfree engine reads θ from the coord and reports best flips back instead of running standalone. Deploy fusedx to CF Pages or Vercel. **Cleanest path.**

Go with #3.

### Changes in `~/Documents/GitHub/fusedx`

Patch `src/lib/gpt-gradfree-engine.ts`:

```typescript
// Top of file
const POSTNET_COORD = process.env.NEXT_PUBLIC_POSTNET_COORD;

// In trainLayerwise (or wherever the main step loop is):
if (POSTNET_COORD) {
  // FEDERATED MODE
  const tickResp = await fetch(`${POSTNET_COORD}/api/tick`, {
    method: "POST",
    body: JSON.stringify({ worker_id: WORKER_ID }),
  });
  const { round, theta_chunk_url, batch_seed } = await tickResp.json();
  // For big models, theta isn't sent inline — coord publishes it to R2/KV
  // and includes a URL. Fetch and decode.
  await this.loadThetaFromUrl(theta_chunk_url);

  // Generate K flip proposals, score each, keep the best
  let best = { indices: [], values: [], delta: 0 };
  for (let trial = 0; trial < TRIALS_PER_REPORT; trial++) {
    const { indices, oldVals, newVals } = this.proposeFlip(K);
    const lossBefore = await this.runForwardLoss(this.wBuf, B);
    // apply flip
    for (let i = 0; i < indices.length; i++) this.W[indices[i]] = newVals[i];
    this.device.queue.writeBuffer(this.wBuf, 0, this.W);
    const lossAfter = await this.runForwardLoss(this.wBuf, B);
    // revert (we don't keep locally — coord decides which to accept globally)
    for (let i = 0; i < indices.length; i++) this.W[indices[i]] = oldVals[i];

    const delta = lossAfter - lossBefore;
    if (delta < best.delta) best = { indices, values: newVals, delta };
  }

  await fetch(`${POSTNET_COORD}/api/tick`, {
    method: "POST",
    body: JSON.stringify({
      worker_id: WORKER_ID,
      round,
      flip_indices: best.indices,
      flip_new_values: best.values,
      delta_loss: best.delta,
    }),
  });
}
```

Coord-side flip-accept loop:

```typescript
private async tick(req: Request): Promise<Response> {
  const body = await req.json<TickBody>();
  this.joined.add(body.worker_id);

  if (body.flip_indices && body.round === this.round) {
    // Track best proposal in this round
    if (!this.bestProposal || body.delta_loss < this.bestProposal.delta) {
      this.bestProposal = {
        worker_id: body.worker_id,
        indices: body.flip_indices,
        values: body.flip_new_values,
        delta: body.delta_loss,
      };
    }
    this.proposalsReceived += 1;
  }

  if (this.proposalsReceived >= TARGET_PROPOSALS) {
    // Apply best
    if (this.bestProposal && this.bestProposal.delta < 0) {
      for (let i = 0; i < this.bestProposal.indices.length; i++) {
        this.theta[this.bestProposal.indices[i]] = this.bestProposal.values[i];
      }
      this.publishThetaChunk();  // upload to R2 / KV, get new URL
    }
    this.round += 1;
    this.bestProposal = null;
    this.proposalsReceived = 0;
  }

  return Response.json({
    round: this.round,
    theta_chunk_url: this.thetaUrl,
    batch_seed: this.currentBatchSeed,
    target: TARGET_PROPOSALS,
  });
}
```

## The R2/KV question

Cloudflare Worker request/response bodies are capped at 100 MB. A 2B BitNet model at 1.58 bits/param = ~400 MB raw. Doesn't fit in one tick response.

Solutions in order of preference:
1. **R2 chunked**: store θ in R2, broadcast new chunks per round (only changed weights). Bandwidth at steady state still tiny because flip changes only K bytes.
2. **KV chunked**: similar, KV has 25 MB value cap so we'd shard across keys.
3. **Workers Cache + Range**: serve θ as a static asset, version it per round.

For the first integration: ship the **delta-only** path. Coord broadcasts only the accepted flip per round. Workers maintain their local θ. Initial θ is loaded from a versioned R2 URL on first connect.

## Phasing

**Phase 1 (1-2 hr):** Tournament protocol with a tiny char-LM (the existing fusedx `/gpt` demo uses TinyShakespeare-class). Get the loop closing — pick best flip, apply, broadcast.

**Phase 2 (2-3 hr):** Move θ to R2 with versioned URLs, delta-only broadcasts. Workers reconstruct θ via accumulated accepted flips since last bootstrap.

**Phase 3 (1-2 hr):** Swap in BitNet 2B via fused-lora. Same protocol, bigger model. The bandwidth claim becomes real.

## What to ship in this session vs. next

This session (today): protocol design complete (this doc), Adam optimizer shipped, demo polished, repo pushed.

Next session:
1. Phase 1 of the integration
2. Verify the protocol works with a tiny model (TinyShakespeare GPT)
3. Decide R2 vs KV for θ storage

Before next session, gather:
- A Cloudflare R2 bucket name + access
- A `wrangler.jsonc` R2 binding example for the coord

## Open questions for next session

1. **Where does training data live?** For a federated demo we want each worker to have its own data shard. For fusedx, that means each browser tab loads a different chunk of TinyShakespeare (or whatever corpus). Easy via `?shard=N` URL param.
2. **How do workers handshake the run config?** Model architecture (`H`, `nLayers`, `vocab`) needs to be agreed on. Coord publishes a `run_config.json` at a fixed URL; workers fetch on connect.
3. **Adam over flip-and-accept?** Could we do a soft form of flip-and-accept where the coord aggregates a probability over which weights to flip, biased by reported `delta_loss`? Research direction; not for v1.
