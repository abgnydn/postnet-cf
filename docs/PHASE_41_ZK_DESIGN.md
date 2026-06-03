# Phase 41 — zk-verified SPSA scalar (design doc + circuit spec)

_Status: **DESIGN ONLY** — no proving code written yet. This doc scopes the
real increment before committing the build. Written 2026-06-03._

Goal: replace postnet's **post-hoc, trust-based** byzantine defense (loss-oracle
audit + magnitude-lie test + no-self-audit rotation) with a **cryptographic
guarantee** that a worker computed its submitted `scalar_g` honestly on a
committed data shard — the VerifBFL idea (arXiv:2501.04319), adapted to
postnet's single-scalar SPSA wire format.

---

## 0. TL;DR / decisions to lock before building

- **Target = the head classifier (Phase 38 MLP), NOT Qwen-NTK.** A faithful zk
  proof requires the loss-defining forward pass *inside the circuit*. The head
  MLP (P = 49 796, two tiny matmuls) is tractable; two Qwen-0.5B forwards are
  ~10⁹⁺ constraints — days of proving, not 81 s. See §1.
- **Only the worker's data is private.** `θ` and the perturbation `u` are
  public (the server already holds θ and reconstructs u from the seed), so the
  expensive RNG (mulberry32 + Box-Muller: `sqrt`/`log`/`cos`/`sin`) stays
  **out of circuit**. See §3. This is the single biggest cost win.
- **Recommended stack: circom + snarkjs (Groth16).** ~200-byte proof, pairing
  verify runs in WASM **inside the Durable Object** in single-digit ms. EZKL
  (Halo2) is lower dev-effort but its verifier is heavier in a Worker. See §5.
- **The attack this actually closes: sybil-id rotation** (PAPER_DRAFT.md:188).
  It does *not* by itself stop a worker from choosing adversarial-but-real
  data. See §2 for the precise guarantee and its limits.
- **Honest effort estimate: ~2–3 weeks for a rigorous end-to-end, not 1.**
  The CLAUDE.md "~1 week / 81 s / 0.6 s" line came from the VerifBFL abstract's
  small-model regime; it's the right ballpark for *verify* and *proof time*,
  but the circuit authoring + fixed-point CE + Merkle commitment + WASM verify
  integration is the bulk of the work. A phased 41a/41b split (§6) de-risks it.

---

## 1. Why Qwen-NTK is out of scope (the feasibility wall)

`scalar_g = (L(θ+εu) − L(θ−εu)) / (2ε)`, where `L` is mean cross-entropy of a
**forward pass over the worker's data batch**. To *prove* `scalar_g` honest,
the two forwards must be arithmetized in-circuit.

| target            | params | MACs / forward | in-circuit feasibility |
|-------------------|--------|----------------|------------------------|
| head MLP (Ph. 38) | 49 796 | ~49.7 K        | tractable (§3)         |
| Qwen2.5-0.5B      | ~494 M | ~10⁹ per token × seq | **infeasible today** |

VerifBFL's reported 81 s proof / 0.6 s verify is for a **small** model in an
IVC/folding scheme; it is *not* a claim that LLM forwards are cheap to prove.
SOTA zkML (DeepProve, JOLT-Atlas) is pushing proving down ~10²–10³×, but a
494 M-param transformer forward is still firmly out of reach for a per-round,
per-worker proof. **We do not attempt Qwen here.** If a future phase wants
flagship-model coverage, the realistic path is the *hybrid commitment* (data
Merkle root + seed-derivation + finite-difference arithmetic, losses left as
committed-but-unproven inputs) — a strictly weaker guarantee, documented as
such. That is explicitly deferred.

---

## 2. Threat model — what zk closes and what it does not

Today's defense (`src/tournament-ntk.ts`):

- **loss-oracle audit** — a trusted/cross worker reports `audit_loss_before`;
  server checks the applied step actually moved loss the claimed direction.
- **magnitude-lie test** — flags `claimedΔ < realΔ − 0.5`.
- **no-self-audit rotation** + **per-worker quarantine** at N frauds.

Documented gap (PAPER_DRAFT.md:188, OPEN_QUESTIONS):
> Quarantine is per-worker-id. An attacker rotating IDs every ~10 wins bypasses
> detection.

### The proven statement (head path)

> "I know a data batch `B` with `commit(B) = C` (a value bound to a
> coordinator-issued shard assignment) such that, for **public** `θ`-commitment
> `Cθ`, **public** perturbation `u` (equivalently its commitment `Cu`), and
> **public** `ε`, evaluating the head MLP forward + mean cross-entropy at
> `θ+εu` and `θ−εu` over `B` yields `L⁺, L⁻` with
> `(L⁺ − L⁻)/(2ε) = scalar_g` (public)."

| attack                                         | post-hoc defense | zk (this design) |
|------------------------------------------------|------------------|------------------|
| fabricate huge negative `claimedΔ`             | caught (mag-lie) | caught (binds scalar_g) |
| report `scalar_g` not derived from any forward | trusted audit    | **impossible**   |
| self-close own audit (Session 5b hole)         | patched          | N/A (no audit)   |
| **sybil id-rotation to dodge quarantine**      | **bypasses**     | **closed** if `C` is bound to a coordinator-issued shard assignment, not worker-chosen |
| compute honestly on adversarially-chosen data  | not addressed    | **NOT closed** — proof says "honest on *this committed* data," not "data is benign." Mitigation stays statistical (shard assignment + outlier rejection). |
| reveal training data                           | n/a              | data stays private (ZK) |

**Bottom line of the guarantee:** zk upgrades "trust the worker's number /
trust the auditor" → "the number provably equals the forward over a committed,
server-assigned shard." It removes the *fabrication* and *sybil-rotation*
classes. It does **not** make data-poisoning go away; that remains a
statistical/aggregation-rule problem (orthogonal, keep Krum-style outlier
rejection on top).

---

## 3. Circuit spec (head MLP)

Public inputs: `Cθ` (Poseidon hash of θ), `Cu` (Poseidon hash of u) **or** `u`
inline, `ε` (fixed-point const), `scalar_g` (fixed-point), `C` (data-shard
commitment / Merkle root). Private witness: the `θ` and `u` preimages, and the
batch `B = {(xₙ ∈ ℝ³⁸⁴, yₙ ∈ {0..3})}`.

Per example `n`, computed **twice** (at `θ⁺=θ+εu` and `θ⁻=θ−εu`):

```
h_j   = b1_j + Σ_i x_i · W1_{i,j}        (D=384 → H=128)   49 152 + 128 MAC
h_j   = ReLU(h_j)                         128 comparisons
o_k   = b2_k + Σ_j h_j · W2_{j,k}        (H=128 → K=4)        512 + 4 MAC
loss += CE(o, y)                          softmax/log over K=4
```

then `scalar_g = (mean(loss⁺) − mean(loss⁻)) / (2ε)`.

### What stays OUT of circuit (the cost wins)

- **mulberry32 + Box-Muller (`u` from seed).** `u` is *public-derivable*; the
  verifier recomputes it from the seed (the DO already has
  `reconstructPerturbation`). Either pass `u` as a public input, or pass `Cu`
  and have the verifier check `Cu == Poseidon(verifier-recomputed u)` outside
  the SNARK. **No transcendental RNG in-circuit.**
- **`θ` integrity.** `θ` is public server state; bind via `Cθ` public input.

### Cost drivers (estimate, to be confirmed by a spike)

- ~50 K MAC × 2 forwards × `|B|` examples. With `|B| = 16`: ~1.6 M MAC.
  Comfortable for Groth16/Halo2 (10⁶–10⁷ constraints).
- **Cross-entropy `exp`/`log` is the only hard nonlinearity.** Fixed-point
  lookup tables (range-checked) — standard but the main authoring + soundness
  risk. K=4 keeps the softmax tiny (4 `exp` + 1 `log` per example per sign).
- **Fixed-point throughout.** Float forward → fixed-point requires choosing a
  scale (e.g. 2⁻¹⁶) and proving the finite-difference within a tolerance band
  `|circuit_scalar_g − reported_scalar_g| < δ`, NOT bit-equality (quantization
  drift vs the JS float forward). Defining `δ` and ensuring it can't be abused
  to smuggle a lie is a **named soundness risk** — needs its own analysis.

---

## 4. Data-shard commitment & sybil binding

The sybil fix only works if `C` is **not** worker-chosen. Sketch:

- Coordinator (the DO) assigns each joining worker a shard range over the
  public `agnews-mini.bin` (or a registered private dataset) and publishes a
  Poseidon-Merkle root per shard at join time.
- Worker's proof must use `C = assigned root`. A rotated id gets a *fresh*
  assignment but cannot reuse a victim's contribution or fabricate, and the
  coordinator can rate-limit assignments per source. Rotation no longer buys a
  free pass through the post-hoc window.
- For a genuinely private dataset, `C` is committed once at registration; the
  proof shows membership without revealing rows.

(Open: assignment must itself be authenticated, else sybil just grabs many
assignments. This pushes toward a registration/stake step — note it, don't
solve it here.)

---

## 5. Toolchain decision matrix

Hard constraint: **verification must run inside a Cloudflare Worker DO** (V8
isolate, WASM ok, no native, tight CPU/time budget). That favors a tiny proof +
cheap verify.

| stack | proof size | verify (WASM/Worker) | dev effort | CE/lookup support | verdict |
|-------|-----------|----------------------|------------|-------------------|---------|
| **circom + snarkjs (Groth16)** | ~200 B | pairing check, ~ms, proven WASM path | high (hand-write circuit) | manual lookups | **recommended** |
| EZKL (Halo2, ONNX→circuit) | larger | heavier verifier in Worker; meant for EVM | **low** (export ONNX) | built-in | good for a *spike*, weak for in-DO verify |
| gnark (Groth16/Plonk, Go) | ~200 B | Go→WASM verify viable | high, Go toolchain | manual | viable alt to circom |
| zkVM (risc0 / SP1) | large | needs wrap to verify cheaply | low (reuse exact loss math in Rust) | trivial (just code) | best *correctness* story, worst proof/verify cost |

**Recommendation:** prototype the forward in **EZKL** first to get a working
proof and a constraint count fast (de-risk the CE lookup), then author the
production circuit in **circom** and verify with **snarkjs** Groth16 in the DO.
snarkjs's `groth16.verify` is the battle-tested WASM verify path; a Groth16
proof + verifying key is small enough to ship to the Worker.

_External grounding (verify before quoting in the paper):_ EZKL/Halo2 proving
is reported up to ~3.7×10² slower than Groth16-based systems for comparable
models, and proof gen can run minutes→hours on non-trivial nets; newer
sumcheck/GKR zkML (DeepProve, JOLT-Atlas) claims ~10³× proving and ~10²×
verify speedups over EZKL. These are vendor/blog numbers — **re-confirm with
our own spike before any paper claim.** Sources:
<https://blog.ezkl.xyz/post/benchmarks/>,
<https://blog.icme.io/the-definitive-guide-to-zkml-2025/>,
<https://github.com/zkonduit/ezkl>, arXiv:2402.02675 (South et al., zkSNARK
model evals), arXiv:2501.04319 (VerifBFL).

---

## 6. Recommended phased build (de-risked)

- **41a — commitment + arithmetic only (no forward in-circuit).** Poseidon
  Merkle data-shard commitment + coordinator shard assignment + a circuit that
  proves the finite-difference `(L⁺−L⁻)/(2ε)=scalar_g` taking `L⁺,L⁻` as
  committed inputs. Closes **sybil rotation**; forward still trusted. Small,
  fast, gives a shippable security win and exercises the DO verify path.
  _~1 week._
- **41b — forward in-circuit (the full guarantee).** Add the MLP+CE forward so
  `L⁺,L⁻` are *proven* from `θ,u,B`, not asserted. This is the real VerifBFL
  equivalent. _~1–2 weeks, gated on the EZKL spike confirming CE-lookup cost._

Each phase is independently valuable and independently publishable.

---

## 7. Concrete next actions (when "go")

1. **EZKL spike (½ day):** export the Phase 38 head MLP to ONNX, wrap the
   two-forward + finite-difference as one graph, run `ezkl` to get a real
   constraint count + proof/verify timings on this exact model. This single
   number decides whether 41b is a week or a month.
2. **DO verify spike (½ day):** stand up `snarkjs.groth16.verify` (or the EZKL
   verifier) inside a throwaway Worker; confirm WASM verify works within the
   isolate CPU budget on a trivial circuit. Verify-in-Worker is the riskiest
   integration assumption — test it before circuit authoring.
3. Only then author 41a's circuit + the wire-format extension (add a
   `proof` blob + `Cθ`/`C` fields to the proposal; today's 20-byte wire stays
   the fast path, proof is an optional adjacent message).

## 8. Open risks (named, unsolved)

- **Fixed-point tolerance band `δ`** (§3) — must be small enough not to smuggle
  a material lie, large enough to absorb honest quantization drift. Needs a
  soundness argument, not a guess.
- **Shard-assignment authentication** (§4) — without it, sybil grabs many
  assignments; pushes toward registration/stake (out of scope here).
- **Verify-in-Worker CPU budget** — unverified until the §7.2 spike.
- **CE lookup soundness** — range-checks must be airtight or the loss can be
  forged. Main circuit-authoring risk.
- **Does NOT stop data poisoning** (§2) — keep statistical aggregation defense.
