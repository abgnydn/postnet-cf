#!/usr/bin/env python3
"""
Generate figures for docs/PAPER_DRAFT.md. Outputs SVG → docs/figures/.
Idempotent; safe to re-run.

Figures:
  - fig-bandwidth.svg     log-log: Adam/flip linear in P vs broadcast-only constant
  - fig-multiseed.svg     5-seed bar chart: final loss + attacker W/F + grow events
  - fig-quarantine.svg    timeline: round at quarantine fire per seed
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Figure 1: Bandwidth scaling, log-log ─────────────────────────────────────
# Data from scripts/bandwidth-sweep.mjs (Phase 22-era):
H        = [32, 128, 512, 2048, 8192]
P        = [129, 513, 2049, 8193, 32769]
adam_kb  = [1.7, 6.4, 24.1, 99.8, 509.0]
flip_kb  = [1.8, 6.5, 24.2, 100.0, 509.2]
spsa_b   = [339, 339, 340, 340, 341]  # broadcast-only ↓/tick

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.loglog(P, [v * 1024 for v in adam_kb], "o-", color="#cc2b2b", label="federated Adam (full θ)")
ax.loglog(P, [v * 1024 for v in flip_kb], "s-", color="#d6a02c", label="flip-and-accept (8K bytes)")
ax.loglog(P, spsa_b,                       "^-", color="#2c7a4a", label="SPSA broadcast-only (~340 B)")
ax.axhline(1024 ** 2 * 100, ls=":", color="gray", alpha=0.6)
ax.text(2e4, 1.5e8, "Cloudflare Workers 100 MB cap", fontsize=7, color="gray", ha="right")
ax.set_xlabel("model parameters $P$ (log scale)")
ax.set_ylabel("per-tick downlink, bytes (log scale)")
ax.set_title("Bandwidth scaling: broadcast-only stays $O(1)$ in $P$")
ax.legend(frameon=False, loc="upper left")
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"fig-bandwidth.{ext}", bbox_inches="tight")
plt.close(fig)
print("✓ wrote", OUT / "fig-bandwidth.svg")


# ── Figure 2: 5-seed multi-seed final state ─────────────────────────────────
seeds   = [1, 2, 3, 4, 5]
losses  = [1.762922, 1.762753, 1.762217, 1.762567, 1.762913]  # final last_loss per seed
etas    = [0.00116, 0.00128, 0.00078, 0.00100, 0.00064]
grows   = [8, 7, 2, 5, 3]
atk_W   = [16, 16, 16, 16, 17]  # attacker wins (all = frauds, rate 1.000)
baseline = 1.7631677

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))

# Final loss per seed, with baseline reference line
x = np.arange(len(seeds))
descent = [baseline - L for L in losses]
ax1.bar(x, descent, color="#2c7a4a", alpha=0.85, width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels([f"seed {s}" for s in seeds])
ax1.set_ylabel(r"loss descent  $L_0 - L_{\mathrm{final}}$")
ax1.set_title("Per-seed loss descent (baseline 1.7632)")
mean_descent = np.mean(descent)
ax1.axhline(mean_descent, ls="--", color="#cc2b2b", lw=1, label=f"mean = {mean_descent:.4f}")
ax1.legend(frameon=False, loc="upper right", fontsize=7)
ax1.grid(True, axis="y", alpha=0.2)

# Attacker W/F (left axis) vs grow events (right axis)
ax2.bar(x, atk_W, color="#cc2b2b", alpha=0.85, width=0.6, label="attacker W = F")
ax2.set_xticks(x)
ax2.set_xticklabels([f"seed {s}" for s in seeds])
ax2.set_ylabel("attacker wins (all flagged fraud)", color="#cc2b2b")
ax2.tick_params(axis="y", labelcolor="#cc2b2b")
ax2.set_title("Byzantine + adaptation per seed")

ax2b = ax2.twinx()
ax2b.spines["top"].set_visible(False)
ax2b.plot(x, grows, "o-", color="#2c4c7a", label="η grow events")
ax2b.set_ylabel("η grow events", color="#2c4c7a")
ax2b.tick_params(axis="y", labelcolor="#2c4c7a")

fig.tight_layout()
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"fig-multiseed.{ext}", bbox_inches="tight")
plt.close(fig)
print("✓ wrote", OUT / "fig-multiseed.svg")


# ── Figure 3: Single-seed loss trajectory (N=1 R=183 run) ───────────────────
# Sampled from the live monitor events (PHASE_40_NEXT6_EMPIRICAL.md, post-quarantine):
R_samples    = [64,  75,  100, 130, 161, 183]
L_samples    = [1.763168, 1.762816, 1.761660, 1.760304, 1.758480, 1.756951]
eta_samples  = [0.00100, 0.00279, 0.00279, 0.00279, 0.00279, 0.00279]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True)

ax1.plot(R_samples, L_samples, "o-", color="#2c7a4a", lw=1.5, ms=5)
ax1.axhline(1.7631677, ls="--", color="gray", alpha=0.6, label="baseline (R=0)")
ax1.set_xlabel("server round R")
ax1.set_ylabel("test loss")
ax1.set_title("Loss descent (N=1, R=183, post-quarantine)")
ax1.legend(frameon=False, loc="upper right", fontsize=7)
ax1.grid(True, alpha=0.2)

ax2.plot(R_samples, eta_samples, "s-", color="#cc6622", lw=1.5, ms=5)
ax2.set_xlabel("server round R")
ax2.set_ylabel("η")
ax2.set_title("η drift via sym-AIMD (21 grow / 0 shrink)")
ax2.grid(True, alpha=0.2)
ax2.axhline(0.001, ls=":", color="gray", alpha=0.6)
ax2.text(64, 0.00102, "init η = 1e-3", fontsize=7, color="gray")

fig.tight_layout()
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"fig-trajectory.{ext}", bbox_inches="tight")
plt.close(fig)
print("✓ wrote", OUT / "fig-trajectory.svg")

print(f"\nAll figures in {OUT}/")
