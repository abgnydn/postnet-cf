# Postnet-CF: Federated LLM gate-controller training across browser tabs with verified byzantine defense

*Working draft v0.6 — not peer reviewed. Zenodo archive: [10.5281/zenodo.20506888](https://doi.org/10.5281/zenodo.20506888) (v0.6); concept DOI [10.5281/zenodo.20506887](https://doi.org/10.5281/zenodo.20506887) always resolves to the latest version.*

## Abstract

We present **Postnet-CF**, a federated learning protocol whose workers are unmodified browser tabs and whose coordinator is a Cloudflare Durable Object. Updates are *single-scalar* SPSA estimates (DeComFL-style); each proposal carries **20 bytes of information** (round, seed, scalar_g, claimed_Δ, audit_loss_before — five 32-bit fields) independent of model size. The reference implementation transports this payload as JSON over HTTP for compatibility with Cloudflare Workers, but a binary encoding would match the theoretical minimum. We adapt the per-round learning rate via a symmetric AIMD rule, giving meaningfully faster loss descent than fixed-η on a 50K-parameter head-classifier task. We then apply the same protocol to **federated gate-controller training on a frozen Qwen2.5-0.5B-Instruct**: K=5000 signed log-gates on the residual streams (NTK-Mirror parameterisation) are trained across browser tabs running ONNX-Runtime-Web, with the base model forward-only and only the K-gate vector evolving over time. A post-apply byzantine defense, hardened with a cross-audit queue and a magnitude-lie test, is validated across 5 random seeds: in 5 of 5 runs, an attacker that fabricates `claimed_delta = −10` is quarantined on the first proposal following its 10th flagged win, at an empirical fraud-detection rate of **1.000 ± 0.000** across all five runs, while the gate vector descends from a baseline loss of 1.7632 to 1.7627 ± 0.0003. The complete system runs on the Cloudflare free tier; a public deployment is available at <https://postnet-cf.abgunaydin94.workers.dev>.

## 1. Background and motivation

Federated learning workers behind home routers, hotel WiFi, and corporate proxies cannot accept inbound connections. Standard FL stacks (Flower, FedML, gRPC-based aggregators) assume reachable workers and require either a heavy bidirectional channel or VPN traversal. **Cloudflare Durable Objects** offer a different shape: every worker establishes its own outbound HTTPS, the coordinator is a single persistent address, and the platform handles routing, retries, and TLS. Workers in any tab on the planet can join via *Open URL* — no install, no port-forwarding, no shared infra.

Mid-2026 frontier-scale FL projects (Nous DisTrO, Pluralis, Prime Intellect's INTELLECT family) target server-side or edge-GPU worker fleets. Their public artifacts have not described browser-tab deployments. Postnet-CF takes the consumer-tab regime as a first-class constraint: anyone with a Chrome tab can contribute a *real* forward pass on a *real* LLM, and the protocol is engineered around that constraint from the bottom up — single-scalar wire format, no-FFT byzantine detection, forward-only parameter surface, free-tier compatible coordinator.

**Contributions.** This work makes three concrete contributions:

1. **A single-scalar federated tournament protocol** whose per-proposal information content is 20 bytes (round + seed + scalar_g + claimed_Δ + audit_loss_before), independent of model size, and whose per-tick downlink (broadcast-only, measured) stays under 400 B at any P up to 1.5 × 10⁹ parameters. The protocol runs on the Cloudflare Workers free tier with a Durable Object as the single persistent coordinator.
2. **A post-apply byzantine defense** combining a no-self-audit queue with a magnitude-lie test, validated end-to-end against an adversarial worker. Across 5 independent seeds, the defense quarantines the attacker on the first proposal following its 10th flagged win, at an empirical fraud-detection rate of 1.000 ± 0.000.
3. **A working application to federated LLM training**: NTK-Mirror gate-controller training on a frozen Qwen2.5-0.5B-Instruct, with the base model forward-only in browser tabs running ONNX-Runtime-Web. Loss descends 1.7632 → 1.7627 ± 0.0003 (multi-seed mean) and 1.7632 → 1.7570 (single-seed extended run R = 183) on the K = 5000-gate vector.

### System overview

```
                        ┌──────────────────────────┐
                        │  Cloudflare Durable Obj. │
                        │       coordinator        │
   ┌─────────────┐      │  ┌────────────────────┐  │      ┌─────────────┐
   │ browser tab │  ↔   │  │ pool (TARGET=2)    │  │   ↔  │ browser tab │
   │ Qwen ONNX   │      │  │ pendingAudits[64]  │  │      │ Qwen ONNX   │
   │ SPSA + θ    │      │  │ θ, η, workerStats  │  │      │ SPSA + θ    │
   └─────────────┘      │  └────────────────────┘  │      └─────────────┘
        ↑                          ↑                            ↑
        │ 20-B proposal            │ broadcast applied flips    │
        │ + audit_loss_before      │ (~340 B/tick, applied_since[])
        ↓                          ↓                            ↓
   ┌─────────────┐                                       ┌─────────────┐
   │ browser tab │  ─────────────  ··· ──────────────►   │ browser tab │
   │  (attacker) │      no-self-audit + magnitude-lie    │   (honest)  │
   └─────────────┘            byzantine defense          └─────────────┘
```

**Figure 1.** Postnet-CF system overview. Every worker is an unmodified browser tab running an ONNX-Runtime-Web forward pass on a frozen base LLM. Workers communicate exclusively with a single Cloudflare Durable Object via outbound HTTPS — no inbound connectivity, no peer-to-peer, no install. Per-round proposals are five 32-bit fields (20 B of information); per-round broadcasts stay under 400 B regardless of model size.

## 2. Protocol layers

The repository ships four tournament variants over a shared substrate. We focus here on the SPSA tournament (Phase 36) and the NTK-Mirror federated-controller variant (Phase 40) — the flip-and-accept and ternary variants from earlier phases are documented in `docs/PROTOCOL.md`.

A coordinator holds `(round, θ, pool, appliedHistory, workerStats)`. Each round, every joined worker:

1. *Polls* the coordinator (`since_round`), gets `applied_since: Flip[]`, replays each `Flip` locally so its `localTheta` matches the coordinator.
2. Runs `T` SPSA *trials* on `localTheta`. Each trial draws a random seed `s`, deterministically reconstructs a Rademacher perturbation `u(s) ∈ ℝ^P`, evaluates the loss at `θ ± ε·u`, and estimates `g = [L(θ+εu) − L(θ−εu)] / 2ε`. The trial's *claimed* contribution is `Δ_claimed = L(θ − η·g·u) − L(θ)`.
3. *Submits* the most-improving trial: `(round, seed, scalar_g, Δ_claimed, audit_loss_before)` — five 32-bit fields, **20 bytes of information content**. The reference implementation encodes this as JSON over HTTP (200–300 B per request); a packed-binary transport would hit the theoretical 20-B minimum.

When the pool holds `TARGET_PROPOSALS` submissions for the current round, the coordinator picks `argmin Δ_claimed`. If it is negative, the coordinator applies `θ ← θ − η · scalar_g · u(seed)` and broadcasts the apply via the per-worker `applied_since` channel. Round advances.

**Per-tick downlink** is dominated by the `applied_since` array and JSON envelope. Static analysis (`scripts/bandwidth-sweep.mjs`) measures the Phase 2 *broadcast-only* protocol — the closest published comparison — at **~340 B per tick**, independent of model size. **Bootstrap** is `O(|θ|)` one-shot, sharded across R2 objects for parameter counts that exceed Cloudflare Workers' 100 MB response cap.

## 3. Adaptive η: symmetric AIMD

The per-round learning rate `η` is adapted by the coordinator after each accepted apply, using the *real* `Δ` observed by the byzantine defense (§5). On every accepted apply with `real_Δ < −threshold`, η is multiplied by `1.05`; on `real_Δ > +threshold`, η is divided by `1.05`. Step bounds `[1e-5, 1e-1]` clamp pathological drift. This is log-symmetric multiplicative AIMD, which lets η drift up monotonically on "honest" downhill applies and pull back instantly on a single uphill apply.

On the Phase 38 head-classifier task (P = 49 796, AG News topic classification on MiniLM features), symmetric AIMD beat fixed-η at R = 90: **loss 1.1235, accuracy 56%**, versus fixed-η's loss 1.3019 / accuracy 40% at R = 100 (data from `docs/PHASE_39_ADAPTIVE_ETA.md`). An Adam-on-scalar variant (Phase 39b, MEAZO-faithful) underperforms sym-AIMD with default hyperparameters because Adam's per-step normalisation caps the effective step magnitude near `lr`; sym-AIMD has no such cap and is allowed to discover its own scale. We adopt sym-AIMD as the canonical η rule for all downstream phases.

## 4. SPSA scaling crossover

A flip-and-accept tournament with `K` index/value updates has a per-tick proposal of `8K` bytes; SPSA carries a fixed 20 B information content per proposal regardless of model size. For small `P`, flip-and-accept descends faster per-round; for large `P`, the upload bandwidth becomes the bottleneck. On the `tournament-lm-big` head (P = 31 707, `docs/PHASE_37_SCALING.md`), SPSA's per-round descent rate matches flip-and-accept at P ≈ 30 000 and dominates beyond. The per-round descent ratio in the phase doc (SPSA 2.8× *improving* across 13× model growth while flip-and-accept degrades 20%) is empirical, not derived from a closed-form scaling law. SPSA becomes the canonical update rule for any model with > 30K trainable parameters.

## 5. Byzantine defense

We adopt the classic "verify, don't trust" pattern. After each apply, the coordinator computes `real_global_delta = L_after − L_before`, where the two loss readings come from a *worker* who is not the one whose proposal was applied (the *no-self-audit* rule, §5.1). A winning proposal with claimed `Δ < −1e-4` but observed `real_global_delta > 1e-4` trips the **inversion test** and is recorded as fraud in the winning worker's stats.

### 5.1 No-self-audit cross-rotation

When the loss oracle was moved client-side (the Cloudflare coordinator cannot host a 500M-parameter Qwen forward), audits became forgeable: an attacker could close its own pending audit with a fabricated `audit_loss_before = 0` and bypass the inversion test. We introduce a queue `pendingAudits[]` of unresolved applies (cap 64), and resolve an incoming `audit_loss_before` from worker X against the *earliest* pending entry whose `winnerId ≠ X`. The server additionally rejects `audit_loss_before ≤ 0` as a sentinel (real cross-entropy is positive). Together these prevent both self-closure and the corruption of the coordinator's loss baseline.

### 5.2 Magnitude-lie test

The inversion test catches an attacker whose fabricated proposal makes loss *go up*. At small ε and small η, however, ~50% of random Rademacher perturbations happen to lower loss; under these conditions the inversion test is silent on the obvious lie that `Δ_claimed = −10` is incompatible with any plausible per-round descent magnitude. We add a second clause: `magnitudeFraud = Δ_claimed < (real_Δ − 0.5)`. Honest per-round Δ at the Phase 40 settings is `O(1e-2)`; the 0.5 margin is conservative.

### 5.3 Tiered quarantine

After at least 10 wins, a worker whose `max(cumulative_rate, last_20_window_rate) > 0.4` (or `last_100_window_rate > 0.25`) is quarantined: subsequent proposals are rejected on submission. Cumulative catches consistent attackers; the sliding window catches *patient attackers* that act honest for the first N wins to dodge the cumulative gate.

### 5.4 Empirical validation (multi-seed)

We ran a 5-seed sweep on Kaggle T4 GPU against the production Cloudflare deployment. Each seed runs one honest Python verifier (Qwen-0.5B forward) and one Python attacker thread that fabricates `claimed_delta = −10`, no audit. R = 100 internal verifier iterations per seed; the coordinator is reset between seeds.

| seed | server R | last_loss | η | grow events | accept rate | attacker W/F | quarantined |
|---|---|---|---|---|---|---|---|
| 1 | 89 | 1.762922 | 0.00116 | 8 | 0.497 | 16 / 16 | yes |
| 2 | 72 | 1.762753 | 0.00128 | 7 | 0.500 | 16 / 16 | yes |
| 3 | 72 | 1.762217 | 0.00078 | 2 | 0.497 | 16 / 16 | yes |
| 4 | 72 | 1.762567 | 0.00100 | 5 | 0.497 | 16 / 16 | yes |
| 5 | 73 | 1.762913 | 0.00064 | 3 | 0.500 | 17 / 17 | yes |
| **mean ± σ** | **75.6 ± 7.4** | **1.762674 ± 0.000294** | **0.00097 ± 0.00026** | **5.0 ± 2.5** | — | — | **rate = 1.000 ± 0.000** |

**Table 1.** Per-seed final state of the 5-seed multi-seed sweep (Kaggle T4, deployed coordinator). The cumulative-rate quarantine gate (`wins ≥ 10 ∧ cumRate > 0.4`) fired in 5/5 runs; the W/F figures exceed 10 because audits continue to land during the network round-trip between gate-trip and rejected proposal.

In **5 of 5 runs**, the attacker's cumulative fraud rate reaches 1.0 on the 10th win (since each of its wins is flagged), making the worker eligible for quarantine; the *very next* proposal it submits is rejected on the cumRate > 0.4 ∧ wins ≥ 10 gate. The empirical fraud-detection rate is **1.000 ± 0.000** — every single audit, every seed, flagged. The W=16/F=16 figure in the table reflects audits that landed during the network round-trip between the 10th win and the next-submission quarantine fire, all of which (correctly) closed at 100% fraud rate without changing the outcome.

![](figures/fig-multiseed.pdf)

**Figure 2.** Multi-seed sweep result. *Left:* per-seed loss descent from baseline (R = 0, L = 1.7632). All five seeds produced near-identical descent magnitude (σ = 2.9 × 10⁻⁴, below per-round Δloss). *Right:* attacker wins (all flagged as fraud, red bars) vs sym-AIMD η-adaptation events (blue line). Detection rate is invariant to seed; η-adaptation cadence varies more because the random walk in θ-space changes how often the AIMD threshold is crossed.

![](figures/fig-trajectory.pdf)

**Figure 3.** Single-seed extended run (N = 1, R = 183, attacker active throughout). *Left:* test loss descends from 1.7632 to 1.7570 (−0.0062), with the descent rate stabilising at ≈ 3.5 × 10⁻⁵ per round after η has reached its self-organised plateau. *Right:* η drifts monotonically 1.0e-3 → 2.8e-3 via 21 sym-AIMD grow events and 0 shrink events — every accepted apply lowered the measured loss past the 1e-5 threshold.

### 5.5 Carryover from char-LM empirical study (Phase 9–14)

The same byzantine defense was previously evaluated on a small char-LM (P = 2 379) with `n = 3` honest workers, `0..3` attackers, and `n_seeds = 3` per cell (Phase 14, prior commit `db3a78c`):

- *Attacker-share scaling*: at 50% attacker share (3 honest + 3 attackers), final loss degraded by only +0.09 nats vs the 0-attacker baseline. The defense's effective tolerance scales with the *count* of attackers it can detect within the 10-win burn-in, not their proportion.
- *No-defense control*: with defense **off**, a 3 honest + 1 byzantine run landed at final loss 2.35 — a +0.72 nat gap vs the 1.63 honest baseline. Enabling the Phase 9–14 defense closed **87% of that gap** to a +0.10-nat residual.

These earlier results provide the only direct evidence in this work that the defense scales with attacker count and quantifies the loss prevented; the §5.4 Qwen-gate sweep is limited to a single attacker by construction. The honest worker's descent is reproducible (`σ_loss < 3e-4`, below the per-round Δ magnitude); the η-adaptation cadence varies more across seeds (`σ_grow ≈ 50% of mean`) because the random walk in θ-space changes how often `real_Δ` exceeds the AIMD threshold — but the *outcome* (loss reached, accept rate, quarantine fire) is invariant to seed.

## 6. Federated LLM gate-controller training

The headline application is **federated training of an NTK-Mirror gate controller on a frozen Qwen2.5-0.5B-Instruct**, across browser tabs. The controller is a sparse set of signed log-gates on residual-stream channels: at decoder layer ℓ and channel c, the hidden state is rescaled `h'_{ℓ,c} = h_{ℓ,c} · m_{ℓ,c}` where `m_{ℓ,c} = exp(MAX_LOG_GATE · tanh(s_{ℓ,c}))`. The `tanh` keeps the multiplier in `[exp(−MAX_LOG_GATE), exp(+MAX_LOG_GATE)]` so a single bad gate cannot destroy the forward pass; `s` is the trainable parameter and `MAX_LOG_GATE` is a per-artifact constant. We select K = 5000 of the 24 × 896 = 21 504 candidate gates by `|∂L/∂s|` on a small math corpus (Chlon, 2026); this gate-selection artifact is a 40,032-byte binary (~40 KB) baked into the coordinator.

A worker holds the Qwen-0.5B ONNX (994 MB int8, served by HuggingFace Hub or an R2 sibling), the gate artifact, and the K = 5000 trainable values `θ ∈ ℝ^K`. Per SPSA trial, the worker injects `exp(MAX_LOG_GATE · tanh(s))` multipliers into the appropriate hidden states via a forward-only ONNX graph surgery and reads off the math-corpus cross-entropy. The SPSA proposal (20 B information content) is submitted to the coordinator; byzantine defense (§5) runs server-side.

In the 5-seed sweep of §5.4, the gate vector descended from baseline `L = 1.7632` to mean `L = 1.7627`; in an earlier single-seed run extended to server R = 183, descent reached `L = 1.7570` with the gate vector's `‖θ‖_2` growing from 0 to 0.634 and per-gate values spanning ±0.034 in log space. The η drifted monotonically `1.0e-3 → 2.8e-3` (21 sym-AIMD grow events, 0 shrinks), confirming that the apply path is dominated by *real* downhill steps even with the attacker active and continuously submitting.

These numbers are modest in absolute terms (K = 5000 trains ~0.001% of Qwen-0.5B's parameters; the math corpus is 4 examples). The contribution is the *system*: a frozen 500M-parameter LLM forward, byzantine-tolerant SPSA gates, browser-tab workers, a free Cloudflare coordinator, a 20-B-information proposal payload, all integrated, all reproducible from a public repo.

## 7. Bandwidth scaling

Per-tick downlink (server's JSON response) across model scales, as measured by `scripts/bandwidth-sweep.mjs`:

| H | P | federated Adam ↓/tick | flip-and-accept ↓/tick | broadcast-only ↓/tick | Bootstrap binary |
|---|---|---|---|---|---|
| 32 | 129 | 1.7 KB | 1.8 KB | 339 B | 524 B |
| 128 | 513 | 6.4 KB | 6.5 KB | 339 B | 2.0 KB |
| 512 | 2 049 | 24.1 KB | 24.2 KB | 340 B | 8.0 KB |
| 2 048 | 8 193 | 99.8 KB | 100.0 KB | 340 B | 32.0 KB |
| 8 192 | 32 769 | 509.0 KB | 509.2 KB | 341 B | 128.0 KB |
| BitNet 2B | 1.5 × 10⁹ | 282 MB (cap-exceeded) | same | 337 B | 282 MB via R2 range read |

**Table 2.** Per-tick downlink across model scales, as measured by `scripts/bandwidth-sweep.mjs`. The federated-Adam and flip-and-accept paths scale linearly in P; the broadcast-only path (used by SPSA) stays under 400 B regardless of model size.

![](figures/fig-bandwidth.pdf)

**Figure 4.** Bandwidth scaling on log–log axes. Both federated Adam (full-θ download) and flip-and-accept (8K-byte index/value pairs) scale linearly in `P` and exceed Cloudflare Workers' 100 MB response cap above `P ≈ 10⁷`. The broadcast-only path used by SPSA stays under 400 B at every measured `P`, because the per-tick payload encodes only the *applied* flip (single seed + scalar_g + bookkeeping), not the model.

Federated Adam and flip-and-accept scale linearly with `P`; both break Cloudflare Workers' 100 MB response cap above `P ≈ 10⁷`. The broadcast-only path — used by SPSA — stays under 400 B regardless of model size, because the per-round payload encodes only the *applied* flips (single seed + scalar_g + small bookkeeping), not the model. The proposal *upload* carries 20 B of information per submission. For the Qwen-0.5B gates application (P = 5000), the one-time bootstrap is the 40 KB gate-selection artifact; subsequent operation is broadcast-only.

## 8. Related work

Evolutionary baselines: the single-winner tournament with parameter "flips" is a federated reformulation of the **(1+1) evolution strategy** (Schwefel, 1981) — a single child is generated per round, accepted if it improves the fitness on the parent population (here, the global θ). Federated learning fundamentals: **FedAvg** (McMahan et al., 2017, arXiv:1602.05629), **Krum / trimmed-mean** (Blanchard et al., 2017, arXiv:1703.02757), the SCAFFOLD / FedProx adaptations of FedAvg. Zero-order federated optimization: **MeZO** (Malladi et al., 2023, arXiv:2305.17333) introduces "memory-efficient zeroth-order optimizer" for LLM fine-tuning; **DeComFL** (Li et al., 2024, arXiv:2405.15861, "Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization") gives the canonical decentralised single-scalar protocol we adapt. Browser-tab and edge swarms: **Pluralis Research** (pluralis.ai) builds model-parallel decentralised "Protocol Learning"; **Nous DisTrO** (Nous Research, 2024, preliminary report at github.com/NousResearch/DisTrO) targets edge GPUs with a distributed-training-over-the-internet optimizer family with ~1 000–10 000× communication reduction; **Prime Intellect's INTELLECT-3** (Nov 2025, 106B-param MoE, GLM 4.5 Air-based) explicitly used a centralised 512×H200 cluster, marking the project's shift from decentralised to centralised training. **NTK-Mirror** gate controllers: Chlon (github.com/leochlon/ntkmirror, May 2026, MIT-licensed, Hassana Labs) introduces the signed log-gate parameterisation we federate. **Verifiable training:** VerifBFL (arXiv:2501.04319) uses zk-SNARKs with incremental verifiable computation to make individual FL workers' contributions cryptographically auditable.

**Prior work by this author:** The Swarm (Günaydın, 2025; see `docs/PHASE_*.md` and the predecessor postnet repository) developed the original 50% byzantine-tolerance result via trimmed-mean over reported gradients; this work attains comparable tolerance at the protocol layer through verified-delta + quarantine.

### 8.1 Direct comparison

To our knowledge, no other published FL protocol combines a single-scalar wire format with a browser-tab deployment and a tested byzantine defense in a single system. The closest entries differ on one or more of these axes:

| System | worker | per-round upload (theoretical min) | byzantine-tolerant | deployable to a browser tab | open + free to reproduce |
|---|---|---|---|---|---|
| **DDP / FedAvg** (McMahan 2017) | server GPU | `O(P)` (full grad / θ) | no (vanilla) | no | yes (code, not infra) |
| **DiLoCo** (DeepMind 2023) | datacenter cluster | `O(P)` outer-loop, infrequent | no | no | partial (Google) |
| **OpenDiLoCo** (Prime Intellect 2024) | A100 cluster | `O(P)` infrequent | no | no | yes (Prime cluster) |
| **DeMo** (Peng 2024) | GPU peer | `O(P/k)` via DCT chunks (k≈100) | no | no | yes (single-script) |
| **DeComFL** (Li 2024) | server peer | `8 B` scalar, but `O(P)` shared-seed setup | no | partially (CPU OK) | yes (code) |
| **INTELLECT-3** (Prime Intellect 2025) | 512×H200 cluster | `O(P)` | no | no | inference weights only |
| **Postnet-CF (this work)** | browser tab | **20 B information** (JSON in current impl) | **yes (no-self-audit + magnitude-lie)** | **yes** | **yes (CF free tier + this repo)** |

**Table 3.** Comparison of recent federated and decentralised training systems on dimensions postnet-cf optimises for. The 20-B information-content figure is the protocol's theoretical minimum (five 32-bit fields per proposal); current implementation transports as JSON over HTTP for Cloudflare Workers compatibility. DeComFL's `8 B` figure is per-scalar but requires a shared-seed bootstrap that scales with model parameter count; postnet-cf's bootstrap is `O(|θ|)` once and then broadcast-only. None of the comparison systems implement an end-to-end byzantine defense as part of the protocol.

## 9. Reproducibility

```bash
git clone https://github.com/abgnydn/postnet-cf
cd postnet-cf
npm install
npx wrangler dev --port 8787
# UI: http://localhost:8787/dashboard.html
# Headless verifiers: scripts/{empirical-study,spsa-verifier,ntk-verifier}.mjs / .py
# Multi-seed sweep (Kaggle/Colab): notebooks/phase40_next6e_sweep_kaggle.ipynb
```

Live deployment: <https://postnet-cf.abgunaydin94.workers.dev>. All code MIT-licensed; protocol spec at `docs/PROTOCOL.md`; per-phase empirical writeups under `docs/PHASE_*.md`.

## 10. Limitations and future work

- **Single-honest degenerate regime.** With exactly one honest worker and the attacker quarantined, the no-self-audit rule prevents the honest peer from closing its own pending audits. The system still progresses (applies happen, loss descends, η adapts) but the byzantine signal goes quiet until a second honest worker joins. We have not empirically tested a swarm of ≥ 2 honest workers post-quarantine; the design intent is that they cross-audit one another's wins, restoring the signal, but this requires a multi-honest sweep we leave to future work.
- **Cross-session sybil resistance.** Quarantine is per-worker-id. An attacker rotating IDs every 10 wins can bypass detection. The path forward is cryptographic data-shard commitments (VerifBFL-style Nova SNARK; arXiv:2501.04319), explored separately.
- **Empirical scale.** Phase 40 results are at K = 5 000 (~0.001% of Qwen-0.5B parameters) on a 4-example math corpus. The protocol scales to wider K and larger base models with no wire changes; a held-out downstream evaluation and base-model sweep remain future work.
- **Multi-seed coverage.** §5.4 is N = 5 against one attack profile (constant-claim spam). Patient attackers, multi-tab coordinated Sybil, and gradient-shape attacks are not in this empirical envelope.

## 11. Acknowledgements

NTK-Mirror gate parameterisation is from Leon Chlon's open-source ntkmirror project (Hassana Labs, May 2026), imported as a Python dependency under MIT licence (attribution in §8, code in `scripts/ntk-verifier.py` and `scripts/extract-ntk-gates.py`). Qwen2.5-0.5B-Instruct base model is © Alibaba Qwen team, used under Apache 2.0 (the int8 ONNX shipped in `~/postnet-cf-onnx/` is a quantized derivative; original weights are not redistributed in this repository). Free coordinator hosting on Cloudflare Workers + Durable Objects. Free GPU sweep runtime on Kaggle.

The postnet-cf repository itself is MIT-licensed (`LICENSE`).

---

*Postnet-CF — A. B. Gunaydin (2026). Working draft v0.6 of a long-running protocol sequence. Zenodo DOI [10.5281/zenodo.20506887](https://doi.org/10.5281/zenodo.20506887) (concept; v0.6 archive [10.5281/zenodo.20506888](https://doi.org/10.5281/zenodo.20506888)). Comments and patches welcome at <https://github.com/abgnydn/postnet-cf>.*
