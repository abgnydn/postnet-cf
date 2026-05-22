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

## Caveats

- The model is small (P = 2 379) and the text is short (340 chars). Real BitNet-class numbers would differ. The point of these numbers is *the protocol's behaviour shape under different worker configurations*, not absolute quality on a real text corpus.
- Variance is across seeds only — there's no separate test-set held out, all scoring is on the same text the model is trained against. "Loss" here is therefore *training loss*, not generalization loss.
- Wall-clock per run depends on which DO is loaded and how busy the local wrangler instance is; numbers should not be compared between machines.
