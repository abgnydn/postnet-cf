# Postnet-CF protocol — v0.5

This document specifies the wire protocol shared by the `Tournament`, `Ternary`, and `TournamentLM` Durable Objects. The federated-Adam coord at `/` uses a separate, simpler protocol (gradient averaging) and is not covered here.

The protocol has three components: **tick** (the request-response cycle), **snapshot** (the bootstrap path), and **WS push** (an optional optimization on top of tick). All three operate on the same `θ` (a model parameter vector) and the same coordinator state.

## State machine (server)

A coord holds:

```
round           : uint32     monotonically increasing; advances when a pool fills
theta           : Float32[]  (or Int8[] for ternary)   the current parameters
pool            : Proposal[] proposals received this round
bestProposal    : Proposal?  the most-negative-delta proposal seen this round
lastLoss        : float      testLoss(θ) — the convergence metric
appliedHistory  : Flip[]     last 1000 flips that were applied
snapshotRound   : uint32     round at which the most recent R2 snapshot was taken
workerStats     : Map<id → {wins, frauds, lastWinRound, recent[20]}>
subscribers     : Set<WebSocket>
```

State transitions are atomic per `tick` request. The only function that advances `round` is the apply phase below.

## Tick request

```jsonc
POST /api/{tournament,ternary,lm}/tick
Content-Type: application/json

{
  "worker_id": "string",         // required; client-generated, stable per session
  "since_round": <uint32>?,      // optional; the client's last known round
  "round": <uint32>?,            // present iff this is a proposal submission
  "indices": <uint32[]>?,        // K param indices to flip
  "values":  <float[] | int[]>?, // K replacement values (int for ternary)
  "delta":   <float>?            // client-claimed Δloss for the proposal
}
```

If `round`/`indices`/`values`/`delta` are absent, the request is a **poll-only** tick: the server returns current state and any pending pushes (via `applied_since`) without admitting a proposal.

If they're present, this is a **proposal submission**: see *Tick handler* below.

## Tick handler (atomic)

```
1. joined.add(worker_id)
2. If proposal fields are present and well-formed:
   a. Check round == server.round  (else: reject as stale)
   b. Check worker not in quarantine
        quarantine if max(cumulative_fraud_rate after >=10 wins,
                          last-20-window fraud_rate) > 0.4
   c. If best for the round so far, become bestProposal
   d. proposalsReceived += 1
3. If proposalsReceived >= TARGET_PROPOSALS:
   a. apply = (bestProposal && bestProposal.delta < 0)
   b. lossBefore = lastLoss
   c. If apply:
      - Mutate θ: for each (idx, val) in bestProposal:
            theta[idx] = val
      - appliedHistory.push({round, indices, values})
      - if length > 1000, shift oldest
   d. lastLoss = testLoss(theta)
   e. If apply, update workerStats[bestProposal.worker_id]:
      - wins += 1
      - real_global_delta = lastLoss - lossBefore
      - is_fraud = (real_global_delta > 1e-4) && (bestProposal.delta < -1e-4)
      - frauds += is_fraud ? 1 : 0
      - recent.push(is_fraud ? 1 : 0); cap at 20
   f. round += 1
   g. pool = []; bestProposal = null; proposalsReceived = 0
   h. Broadcast to every WebSocket in subscribers:
        { type: "advance", round, last_loss, applied: bestProposal ?? null }
4. Compute applied_since = appliedHistory.filter(f => f.round >= since_round)
5. Return response (see below)
```

## Tick response

```jsonc
{
  "round":          <uint32>,
  "P":              <uint32>,                    // model size constant
  "last_loss":      <float>,
  "last_applied":   <Flip | null>,               // the flip applied THIS tick (null if none)
  "applied_since":  <Flip[] | null>,             // all flips since `since_round`, or null if missing
  "oldest_applied_round": <uint32 | null>,       // for drift detection
  "accepted":       <bool>,                       // server admitted this submission to the pool
  "rejected":       <bool>,                       // round mismatch (stale)
  "quarantined":    <bool>,                       // worker is in the byzantine penalty box
  "advanced":       <bool>,                       // round incremented as a result of this tick
  // protocol-specific:
  "task":          <"wave" | "circle" | "xor">,  // (tournament, ternary; not lm)
  "scale":         <float>,                      // (ternary only)
  "V", "E", "HID", "CTX":                        // (lm only) architecture constants
  "flip_size":     <uint32>,
  "target":        <uint32>,                     // TARGET_PROPOSALS = 2
  "proposals":     <uint32>,                     // count this round so far
  "joined":        <uint32>,
  "accept_rate":   <float>                       // accepted / considered, lifetime
}
```

`Flip` = `{ round: uint32, indices: uint32[], values: number[] }`.

## Snapshot manifest

```
GET /api/{lm}/snapshot
→ {
    "round":             <uint32>,        // round at which the snapshot was taken
    "P":                 <uint32>,
    "shards":            <ShardRef[]>,    // (lm only — older endpoints return a single blob)
    "num_shards":        <uint32>,
    "shard_size_floats": <uint32>,
    "snapshot_bytes_total": <uint32>
  }

ShardRef = {
  "url":         "/api/lm/snapshot.bin?round=N&shard=K",
  "shard":       <uint32>,
  "float_start": <uint32>,
  "float_count": <uint32>,
  "bytes":       <uint32>      // = (shard==0 ? 8 : 0) + float_count * 4
}
```

The simpler tournament + ternary endpoints return a single `snapshot_url`:

```
GET /api/{tournament,ternary}/snapshot
→ {
    "round":           <uint32>,
    "task":            <Task>,
    "P":               <uint32>,
    "snapshot_url":    "/api/{tournament,ternary}/snapshot.bin?round=N",
    "snapshot_bytes":  <uint32>,
    "scale":           <float>      // ternary only
  }
```

## Snapshot bytes

Tournament (float weights):
```
[uint32 LE round][uint32 LE P][P × float32 LE]    total: 8 + P*4 bytes
```

Ternary:
```
[uint32 LE round][uint32 LE P][float32 LE scale][ceil(P*2/8) bytes packed]
```
Each ternary value packs into 2 bits: `00 = 0`, `01 = +1`, `10 = -1`. So a P=129 ternary snapshot is `12 + 33 = 45 B`; a P=1.5B ternary snapshot is `12 + ~375 MB`.

LM sharded:
```
shard 0:    [uint32 round][uint32 P][FLOATS_PER_SHARD × float32]    total: 8 + 4 * count
shard k>0:                            [FLOATS_PER_SHARD × float32]    total: 4 * count
```
The header lives only in shard 0; subsequent shards are raw float32 payloads, contiguous slices of θ.

## WebSocket push (optional)

```
GET /api/{tournament,ternary,lm}/ws        Upgrade: websocket
```

On accept, server sends one `hello`:
```
{ "type": "hello", "round": <N>, "last_loss": <float>, "recent": <Flip[]> }
```
where `recent` is the last ≤ 50 entries from `appliedHistory`.

Thereafter, every round advance broadcasts:
```
{ "type": "advance", "round": <N>, "last_loss": <float>, "applied": <Flip | null> }
```

Clients should treat push as best-effort: if the WebSocket closes, fall back to polling `/tick` with `since_round` to recover any missed flips. The `appliedHistory` window (1000 entries) means a client can be disconnected for thousands of rounds and still catch up without re-bootstrapping, provided its `localRound` is still within the window.

## Threat model

Honest worker: scores proposals on its local batch (private shard or full text), submits its best `(indices, values, delta)`. Each `delta` is the worker's *estimate* of the global loss change from applying that flip; it may be biased by shard locality but signs should usually agree with the global delta.

Byzantine worker: submits arbitrary `(indices, values)` with an arbitrarily-negative `delta` to win the tournament regardless of merit.

Defense: post-apply, server computes `real_global_delta = lastLoss_after - lastLoss_before`. If `real_global_delta > 1e-4` while the winner's `delta < -1e-4`, mark this win as a fraud. After 10 wins, if `max(cumulative, last_20_window) > 0.4`, the worker is quarantined — its future proposals are dropped before they enter the pool.

Limitations:
- The defense is per-worker. A coordinated attack with many fresh worker IDs (each below the 10-win gate) can dodge detection. Mitigation would need cross-session reputation tied to a stable identifier (IP, DKIM signature, capability token).
- The "patient attacker" (act honest 9 wins, then attack) is caught by the 20-win sliding window, but a more patient attacker (act honest 50 wins, then attack briefly, return to honest) could still slip flips through. The defense protects against sustained sabotage; brief opportunistic attacks are partly viable.

## Reference implementations

All implementations in this repository:

| component | tournament | ternary | char-LM |
|---|---|---|---|
| server DO | `src/tournament.ts` | `src/ternary.ts` | `src/tournament-lm.ts` |
| browser worker | `public/tournament-worker.js` | `public/ternary-worker.js` | `public/lm-worker.js` |
| headless verifier | `scripts/tournament-verifier.mjs` | `scripts/ternary-verifier.mjs` | `scripts/lm-verifier.mjs` |

Empirical-study driver: `scripts/empirical-study.mjs` (supports the LM coord under three modes: variants, attackers, smart).

Static bandwidth analyzer: `scripts/bandwidth-sweep.mjs` (no live coord needed).

## Versioning

`v0.5` — this document. Captures protocol after phases 1-19.

Field-level back-compat: clients that don't send `since_round` get `applied_since: null` and must re-bootstrap on any round mismatch. Clients that ignore unknown fields (the dashboard, third-party tools) continue to work as the protocol grows.

Breaking changes go through a major version bump and a temporary dual-routing window. None planned at this writing.
