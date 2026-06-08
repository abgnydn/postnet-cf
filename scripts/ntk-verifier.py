#!/usr/bin/env python3
"""
Phase 40 next-2 — federated NTK-Mirror verifier (Python, Qwen forward via ntkmirror).

One Python process = one federated worker. Polls /api/ntk/tick to get the
current round, runs 4 SPSA trials per round through Qwen-0.5B (with our
gates injected via ntkmirror's hook), submits (seed, scalar_g, claimed_Δ).

Acceptance: server's theta evolves over rounds, verifier-side test loss
measurably decreases.

Why Python (not Node):
  - The DO (Cloudflare Worker) can't load a 500M-param model, so server-side
    test-loss / byzantine check is deferred to Phase 40 next-3.
  - ntkmirror's hook-based gate application is much easier to drive in
    Python than to re-implement on top of an ONNX-runtime + Transformers.js
    bundle. The browser worker (next-3) WILL re-implement; this verifier
    exists so we can validate the protocol layer end-to-end NOW.

Run via the same venv used by scripts/extract-ntk-gates.py:

  ~/ntkmirror/.venv/bin/python scripts/ntk-verifier.py \\
      --coord http://localhost:8787 \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --train ~/ntkmirror/examples/math_train.jsonl \\
      --artifact public/data/qwen05b-math-gates-k5000.bin \\
      --rounds 50 --trials 4

CRITICAL: the perturbation reconstruction (mulberry32 + Box-Muller) MUST be
byte-identical to TS-side `reconstructPerturbation` in src/tournament-ntk.ts
(and identical to all prior postnet SPSA tournaments since Phase 36).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import urllib.request
import urllib.error

from transformers import AutoModelForCausalLM, AutoTokenizer
from ntkmirror.controller import _SignedLogMaskModule
from ntkmirror.data import Example, batches, make_batch
from ntkmirror.layers import find_decoder_layers
from ntkmirror.losses import causal_loss_from_logits


# ─── postnet artifact format (mirrors src/ntk-gate.ts) ───────────────────────

MAGIC = b"NTKG"
HEADER_FMT = "<4sIIIIfQ"
HEADER_SIZE = 32


def parse_gate_artifact(path: Path) -> dict:
    buf = path.read_bytes()
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"artifact too small: {len(buf)} < {HEADER_SIZE}")
    magic, version, K, n_layers, hidden_size, max_log_gate, model_hash = \
        struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    expected = HEADER_SIZE + K * 2 + K * 2 + K * 4
    if len(buf) < expected:
        raise ValueError(f"artifact truncated: {len(buf)} < {expected} for K={K}")
    off = HEADER_SIZE
    layer_idx = np.frombuffer(buf[off:off + K * 2], dtype="<u2").copy()
    off += K * 2
    channel_idx = np.frombuffer(buf[off:off + K * 2], dtype="<u2").copy()
    off += K * 2
    raw = np.frombuffer(buf[off:off + K * 4], dtype="<f4").copy()
    return {
        "K": int(K), "n_layers": int(n_layers), "hidden_size": int(hidden_size),
        "max_log_gate": float(max_log_gate), "model_hash": int(model_hash),
        "layer_indices": layer_idx, "channel_indices": channel_idx, "raw_init": raw,
    }


# ─── mulberry32 + Box-Muller (matches src/ntk-gate.ts → TS reconstruction) ───

def reconstruct_perturbation(seed: int, P: int) -> np.ndarray:
    """Bit-identical to TS reconstructPerturbation() — must NOT diverge."""
    t = seed & 0xFFFFFFFF
    out = np.zeros(P, dtype=np.float32)
    # mulberry32 generator
    def rng():
        nonlocal t
        t = (t + 0x6D2B79F5) & 0xFFFFFFFF
        r = t
        # JS Math.imul: 32-bit signed multiply truncated
        def imul(a, b):
            return ((a * b) & 0xFFFFFFFF if (a * b) >= 0 else (((a * b) + 0x100000000) & 0xFFFFFFFF))
        r = imul(r ^ (r >> 15), r | 1)
        r ^= (r + imul(r ^ (r >> 7), r | 61)) & 0xFFFFFFFF
        return ((r ^ (r >> 14)) & 0xFFFFFFFF) / 4294967296.0
    i = 0
    while i < P:
        u1 = 0.0
        while u1 == 0.0:
            u1 = rng()
        u2 = 0.0
        while u2 == 0.0:
            u2 = rng()
        mag = np.sqrt(-2.0 * np.log(u1))
        ang = 2.0 * np.pi * u2
        out[i] = float(mag * np.cos(ang))
        if i + 1 < P:
            out[i + 1] = float(mag * np.sin(ang))
        i += 2
    return out


# ─── HTTP helpers (postnet wire format) ──────────────────────────────────────
# UA spoofs a real browser. Cloudflare's default bot detection 403s the
# Python-urllib/* User-Agent, especially when the request originates from
# a data-center IP (Colab, Vercel, etc.).
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/126.0.0.0 Safari/537.36 postnet-ntk-verifier/1.0")


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def http_post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"content-type": "application/json",
                                           "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def bootstrap_theta(coord: str, expected_K: int) -> tuple[int, np.ndarray]:
    """Fetch the current snapshot and assemble θ = raw[K] of length expected_K."""
    meta = http_get_json(f"{coord}/api/ntk/snapshot")
    if meta["K"] != expected_K:
        raise ValueError(f"server K={meta['K']} != artifact K={expected_K}")
    shards = meta["shards"]
    theta = np.zeros(expected_K, dtype=np.float32)
    round_num = 0
    for s in shards:
        buf = http_get_bytes(f"{coord}{s['url']}")
        header_size = 8 if s["shard"] == 0 else 0
        if s["shard"] == 0:
            # First 8 bytes = uint32 round, uint32 P
            round_num = int.from_bytes(buf[0:4], "little")
        floats = np.frombuffer(buf[header_size:], dtype="<f4")
        start = s["float_start"]
        theta[start:start + len(floats)] = floats
    return round_num, theta


# ─── verifier ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord", default="http://localhost:8787")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--worker-id", default=None,
                    help="defaults to ntk-py-{pid}-{random}")
    ap.add_argument("--device", default="mps",
                    help="cpu | mps | cuda (mps recommended for M-series)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval", type=Path, default=None,
                    help="held-out eval JSONL (disjoint from --train). When set, "
                         "eval loss at the current θ is measured every round and "
                         "written to --trajectory. A monotone eval-loss drop is the "
                         "generalisation signal (Phase 40 learning run).")
    ap.add_argument("--eval-batch-size", type=int, default=None,
                    help="how many eval examples to use (default: all in --eval)")
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="write per-round (round,train_loss,eval_loss,eta) CSV here")
    ap.add_argument("--reset", action="store_true", help="reset server state on startup")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed numpy + torch for reproducible SPSA seed selection (Phase 40 next-6e multi-seed sweep)")
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"seed:      {args.seed}")

    worker_id = args.worker_id or f"ntk-py-{int(time.time() * 1000) & 0xFFFF}"
    print(f"worker_id: {worker_id}")
    print(f"coord:     {args.coord}")
    print(f"model:     {args.model}")
    print(f"artifact:  {args.artifact}")
    print(f"rounds:    {args.rounds}  trials: {args.trials}  ε={args.epsilon}  η={args.eta}")
    print()

    # ── 0. parse artifact ────────────────────────────────────────────────────
    art = parse_gate_artifact(args.artifact)
    K = art["K"]
    print(f"artifact:  K={K}  layers={art['n_layers']}  hidden={art['hidden_size']}  "
          f"max_log_gate={art['max_log_gate']}  hash=0x{art['model_hash']:016x}")

    # ── 1. server reset + bootstrap ──────────────────────────────────────────
    if args.reset:
        http_post_json(f"{args.coord}/api/ntk/reset", {})
        print("server reset")
    round_num, theta = bootstrap_theta(args.coord, K)
    print(f"bootstrap: round={round_num}  ||θ||={float(np.linalg.norm(theta)):.4f}")

    # ── 2. load model + tokenizer ────────────────────────────────────────────
    device = torch.device(args.device)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    print(f"loading {args.model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype_map[args.dtype], device_map={"": device.type})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_path, decoder_layers = find_decoder_layers(model)
    if len(decoder_layers) != art["n_layers"]:
        raise ValueError(
            f"layer count mismatch: model has {len(decoder_layers)}, "
            f"artifact says {art['n_layers']}")

    # ── 3. attach NTK-Mirror controller with our raw values ─────────────────
    ctrl = _SignedLogMaskModule(
        decoder_layers,
        torch.from_numpy(art["layer_indices"].astype(np.int64)),
        torch.from_numpy(art["channel_indices"].astype(np.int64)),
        hidden_size=art["hidden_size"],
        max_log_gate=art["max_log_gate"],
        raw_init=torch.from_numpy(theta.astype(np.float32)),
    ).to(device)
    ctrl.attach()

    # ── 4. load training examples ────────────────────────────────────────────
    examples: list[Example] = []
    with args.train.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(Example(prompt=obj["prompt"], completion=obj["completion"]))
    print(f"loaded {len(examples)} training examples")

    # We'll reuse the same single batch for all forwards in a round (deterministic
    # per-round shard; matches the "private worker shard" pattern from earlier phases).
    batch = make_batch(tok, examples[:args.batch_size], device=device, max_length=args.max_length)

    # Held-out eval batch (Phase 40 learning run). Loaded from a JSONL that is
    # disjoint from --train; loss here is never optimised against, so a drop is
    # genuine generalisation rather than memorisation of the train batch.
    eval_batch = None
    if args.eval is not None:
        eval_examples: list[Example] = []
        with args.eval.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                eval_examples.append(Example(prompt=obj["prompt"], completion=obj["completion"]))
        ebs = args.eval_batch_size or len(eval_examples)
        eval_batch = make_batch(tok, eval_examples[:ebs], device=device, max_length=args.max_length)
        print(f"loaded {len(eval_examples)} held-out eval examples (using {ebs})")

    @torch.no_grad()
    def loss_on(raw_np: np.ndarray, b: dict) -> float:
        # Copy custom raw values into the controller's nn.Parameter; ctrl.s is a
        # property that recomputes from raw every forward.
        ctrl.raw.data.copy_(torch.from_numpy(raw_np.astype(np.float32)).to(device))
        out = model(input_ids=b["input_ids"],
                    attention_mask=b.get("attention_mask"),
                    use_cache=False)
        return float(causal_loss_from_logits(out.logits, b["labels"]).item())

    def loss_with_raw(raw_np: np.ndarray) -> float:
        return loss_on(raw_np, batch)

    base_loss = loss_with_raw(theta)
    print(f"loss @ bootstrap (θ all zeros): {base_loss:.4f}")
    print()

    # ── 5. main loop ─────────────────────────────────────────────────────────
    history = []      # (round, loss_before)
    # Track server's current η; it adapts via Phase 39 sym-AIMD once audited
    # loss starts arriving (next round's first audit closes the loop on this
    # round's winner). Start from CLI value; pull from /tick responses.
    current_eta = float(args.eta)
    t_start = time.time()
    for it in range(args.rounds):
        # poll for current round (handles other workers' applied flips)
        pulled = http_post_json(f"{args.coord}/api/ntk/tick",
                                {"worker_id": worker_id, "since_round": round_num})
        if isinstance(pulled.get("eta"), (int, float)):
            current_eta = float(pulled["eta"])
        # reconcile applied_since
        if pulled.get("applied_since"):
            for flip in pulled["applied_since"]:
                if flip["round"] < round_num:
                    continue
                u_flip = reconstruct_perturbation(int(flip["seed"]), K)
                # Phase 40 next-7: replay with the EXACT η the server applied
                # this flip with (flip["eta"]), not our current η — otherwise
                # replicas drift and the cross-worker audit false-quarantines.
                flip_eta = float(flip.get("eta", current_eta))
                theta -= flip_eta * float(flip["scalar_g"]) * u_flip
        round_num = int(pulled["round"])

        # local loss at current θ — this is "loss_before" for THIS round's
        # claimed_Δ, AND the trusted-auditor signal the server uses to
        # compute real_Δ for the previous round's winner.
        loss_before = loss_with_raw(theta)
        eval_loss = loss_on(theta, eval_batch) if eval_batch is not None else None
        history.append((round_num, loss_before, eval_loss, current_eta))

        # K SPSA trials, all using server's CURRENT η for the trial step.
        best = None
        for t in range(args.trials):
            seed = (np.random.randint(0, 2**32 - 1)) & 0xFFFFFFFF
            u = reconstruct_perturbation(int(seed), K)
            theta_plus = theta + args.epsilon * u
            theta_minus = theta - args.epsilon * u
            loss_plus = loss_with_raw(theta_plus)
            loss_minus = loss_with_raw(theta_minus)
            g = (loss_plus - loss_minus) / (2.0 * args.epsilon)
            theta_step = theta - current_eta * g * u
            loss_at = loss_with_raw(theta_step)
            delta = loss_at - loss_before
            if best is None or delta < best["delta"]:
                best = {"seed": int(seed), "scalar_g": float(g), "delta": float(delta)}

        # submit — include audit_loss_before so the server can close the
        # byzantine + sym-AIMD loop on the PRIOR round's winner.
        reported = http_post_json(f"{args.coord}/api/ntk/tick", {
            "worker_id": worker_id,
            "round": round_num,
            "seed": best["seed"],
            "scalar_g": best["scalar_g"],
            "delta": best["delta"],
            "since_round": round_num,
            "audit_loss_before": float(loss_before),
        })
        if isinstance(reported.get("eta"), (int, float)):
            current_eta = float(reported["eta"])

        # if server advanced, apply the accepted flip locally so we match
        # the server.
        if reported.get("advanced") and reported.get("last_applied"):
            f = reported["last_applied"]
            if f["round"] == round_num:    # the round that just advanced
                u_flip = reconstruct_perturbation(int(f["seed"]), K)
                # Phase 40 next-7: replay with the flip's own η (server-stamped),
                # not current η — keeps this replica bit-identical to the server.
                flip_eta = float(f.get("eta", current_eta))
                theta -= flip_eta * float(f["scalar_g"]) * u_flip
                round_num = int(reported["round"])
        else:
            round_num = int(reported["round"])

        if (it + 1) % 5 == 0 or it == 0:
            elapsed = time.time() - t_start
            grow = reported.get("eta_grow_events", "?")
            shr = reported.get("eta_shrink_events", "?")
            quar = " QUAR" if reported.get("quarantined") else ""
            eval_str = f"  eval={eval_loss:.4f}" if eval_loss is not None else ""
            print(f"  it={it+1:4d}  server_r={round_num:4d}  "
                  f"loss_before={loss_before:.4f}{eval_str}  best_Δ={best['delta']:+.4f}  "
                  f"η={current_eta:.2e}  grow={grow} shr={shr}  "
                  f"||θ||={float(np.linalg.norm(theta)):.4f}{quar}  ({elapsed:.1f}s)")

    # ── 6. final eval ────────────────────────────────────────────────────────
    final_loss = loss_with_raw(theta)
    print()
    print(f"start train loss:  {base_loss:.4f}")
    print(f"final train loss:  {final_loss:.4f}")
    print(f"train Δ:           {final_loss - base_loss:+.4f}")
    if eval_batch is not None:
        # Held-out generalisation summary. base/final measured at θ_bootstrap
        # (all-zeros gates = base model) and θ_final.
        eval_start = history[0][2] if history else None
        eval_final = loss_on(theta, eval_batch)
        if eval_start is not None:
            print(f"start eval  loss:  {eval_start:.4f}  (held-out, never trained on)")
            print(f"final eval  loss:  {eval_final:.4f}")
            print(f"eval  Δ:           {eval_final - eval_start:+.4f}  "
                  f"{'← generalises' if eval_final < eval_start else '← no held-out gain'}")
    print(f"||θ_final||: {float(np.linalg.norm(theta)):.4f}")
    print(f"max |θ_i|:   {float(np.max(np.abs(theta))):.4f}")

    # ── 7. trajectory dump (for plotting train vs eval descent) ───────────────
    if args.trajectory is not None:
        import csv
        with args.trajectory.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "train_loss", "eval_loss", "eta"])
            for r, tl, el, et in history:
                w.writerow([r, f"{tl:.6f}", "" if el is None else f"{el:.6f}", f"{et:.6g}"])
        print(f"trajectory: {args.trajectory} ({len(history)} rows)")

    return 0 if final_loss < base_loss else 1


if __name__ == "__main__":
    sys.exit(main())
