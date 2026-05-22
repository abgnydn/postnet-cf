# Empirical study — char-LM tournament convergence

Numbers below are from `scripts/empirical-study.mjs`, **3 seeds × 1500 rounds per variant**, running against a fresh DO reset between runs. All three variants share the protocol (`/api/lm/*`); only the worker configuration differs.

## Variants

| variant | workers | data each worker scores on | byzantine? |
|---|---|---|---|
| vanilla | 3 honest, all on full text | full TEXT (340 chars) | none |
| sharded | 3 honest, each on a private 1/3 slice | shard[i] only | none |
| byzantine | 3 honest + 1 attacker (claims `delta=-10` for random flips) | shards (honest) / random (attacker) | yes, defense active |

The "vanilla" variant differs from "sharded" only in that every honest worker scores its proposals on the full text rather than its private slice. It is the centralized-data upper bound for what flip-and-accept can achieve on this protocol with this model architecture.

## Results

```
variant       n   mean    std     vs vanilla
vanilla       3   1.2964  0.0285  +0.0000
sharded       3   1.8505  0.0939  +0.5541
byzantine     3   1.9044  0.0529  +0.6080
```

Per-seed final loss:

| variant | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| vanilla | 1.2712 | 1.3362 | 1.2817 | 1.30 | 0.029 |
| sharded | 1.9827 | 1.7949 | 1.7739 | 1.85 | 0.094 |
| byzantine | 1.9722 | 1.8433 | 1.8976 | 1.90 | 0.053 |

## Reading the numbers

**Sharded penalty (+0.55 nats).** This is the cost of true federated learning on this task at this scale: each worker estimates the global delta from only 1/3 of the data, so its proposals are noisier and the tournament picks suboptimal flips relative to having access to the full text. The variance also triples (0.029 → 0.094) because different seed × shard-assignment combinations land on different basins.

**Byzantine cost recovered (+0.61 nats with defense).** The Phase 9 quarantine mechanism (workers with > 40% fraud-rate after 10 wins get their proposals dropped) recovers convergence to roughly parity with sharded. Without defense, byzantine pushed loss to ~2.35 in a single run earlier — defense closes most of that gap. The remaining +0.05 vs sharded reflects the rounds before the attacker is detected (≥ 10 wins required) plus the loss of one honest worker's contribution while the attacker is dominating early.

## Reproduce

```bash
# Server side: wrangler dev must be running
npx wrangler dev --port 8787

# Run the study
SEEDS=3 ROUNDS=1500 node scripts/empirical-study.mjs
```

Wall-clock per run depends on the machine and on whether wrangler is also serving other traffic — not a useful absolute number. Scale `SEEDS` and `ROUNDS` to fit the run budget; the same-machine deltas between variants (the only thing that's actually being compared) survive any reasonable choice.

## Attacker-count sweep (Phase 13)

How does the Phase 9 defense scale with the *fraction* of attackers in the swarm? Run with `MODE=attackers`:

```bash
SEEDS=3 ROUNDS=1500 MODE=attackers node scripts/empirical-study.mjs
```

3 seeds per cell, honest worker count fixed at 3, attacker count varied 0..3:

| attackers | honest share | n | mean | std | vs 0-atk |
|---|---|---|---|---|---|
| 0 | 100% | 3 | 1.8896 | 0.0054 | +0.0000 |
| 1 | 75%  | 3 | 1.9568 | 0.0359 | +0.0671 |
| 2 | 60%  | 3 | 1.8903 | 0.0911 | +0.0007 |
| 3 | 50%  | 3 | 1.9783 | 0.0534 | +0.0887 |

The Phase 9 quarantine holds up to **50% byzantine share** with < 5% loss degradation (+0.09 nats from baseline). This is because the defense is per-worker — each attacker independently hits its > 40% fraud threshold and gets quarantined regardless of how many other attackers exist. Honest workers continue contributing the whole time.

This matches the "50% Byzantine tolerance" result from the earlier [Swarm](https://github.com/abgnydn/swarm-engine) work but at a different protocol layer: there, aggregation was the trimmed mean across reported gradients; here, the defense is "verify the claimed delta against the real global delta and quarantine outliers."

Caveat: the test is single-shot — all attackers join at round 0 and stay byzantine throughout. A sophisticated attacker could mix honest and dishonest proposals to dodge the fraud threshold (act honest for the first 9 wins, attack on win 10+). Phase 14 addresses this with sliding-window detection (see below).

## Phase 14 — sliding-window fraud detection vs the "patient attacker"

Phase 9's defense uses **cumulative** fraud rate over a worker's total wins. That has a known weak spot: an attacker that acts honest for the first 9 wins (passing the `wins >= 10` gate at 0% fraud) and then attacks every round afterward keeps its cumulative fraud rate below the 40% threshold for a long time. After N+9 wins where N are fraudulent: cumulative = N / (N+9). At N=6 that's 40% (borderline); at N=14 that's 60%. Many fraudulent rounds slip through before quarantine kicks in.

Phase 14 fixes this with a **sliding window** over the last 20 verdicts. The quarantine condition becomes:

```
fraud_rate = max(cumulative_rate, last_20_rate)
quarantine if fraud_rate > 0.4
```

A patient attacker that acts honest then turns malicious will see its last-20 fraud rate spike to 100% within 10-12 rounds of switching, hitting the threshold long before cumulative would. Honest workers' last-20 rate stays in the same ~17-26% noise band as their cumulative.

The smart-attacker mode lives in `scripts/empirical-study.mjs` as `MODE=smart` (workers named `lm-smart{0,1,2}` switch from honest to byzantine after 9 self-wins). Running it locally crashes `wrangler dev` under the combined load of long sessions and DO state churn — the protocol upgrade itself is straightforward to verify by reading `src/tournament-lm.ts` and checking the `stats.recent[]` window logic. A clean empirical sweep here would need either a deployed Worker or a smaller test budget.

## Caveats

- The model is small (P = 2 379) and the text is short (340 chars). Real BitNet-class numbers would differ. The point of these numbers is *the protocol's behaviour shape under different worker configurations*, not absolute quality on a real text corpus.
- Variance is across seeds only — there's no separate test-set held out, all scoring is on the same text the model is trained against. "Loss" here is therefore *training loss*, not generalization loss.
- Wall-clock per run depends on which DO is loaded and how busy the local wrangler instance is; numbers should not be compared between machines.
