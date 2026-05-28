#!/usr/bin/env python3
"""
Phase 40 — extract top-K NTK-Mirror gates for a (base_model, corpus) pair
and dump them as a postnet-cf binary artifact.

Run this ONCE per (base_model, task) pair. The output is a static asset under
public/data/ that ships with the worker; the federated SPSA tournament then
only updates the K trainable scalars (raw[K]) per round.

Usage:
  ./scripts/extract-ntk-gates.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --train ~/ntkmirror/examples/math_train.jsonl \\
      --gates 5000 \\
      --out public/data/qwen05b-math-gates-k5000.bin

This script lives in postnet-cf but uses the venv we installed at
~/ntkmirror/.venv. Run via:

  ~/ntkmirror/.venv/bin/python ./scripts/extract-ntk-gates.py ...

Binary artifact format (little-endian throughout):

  HEADER (32 B):
    [ 0:  4]  u32  magic         = 0x4E_54_4B_47  ('NTKG' big-endian — we write
                                                   it as bytes b"NTKG" so it
                                                   reads as the four ASCII
                                                   chars in any hex dump)
    [ 4:  8]  u32  version       = 1
    [ 8: 12]  u32  K             (number of selected gates)
    [12: 16]  u32  n_layers      (decoder layers in the base model)
    [16: 20]  u32  hidden_size   (channels per residual stream)
    [20: 24]  f32  max_log_gate  (0.05 default — bound on |s|)
    [24: 32]  u64  model_id_hash (FNV-1a 64-bit of model_name string,
                                  for sanity-checking worker ↔ artifact match)

  BODY:
    layer_indices  : K × u16   (2*K B)   — selected decoder-layer id per gate
    channel_indices: K × u16   (2*K B)   — selected channel id per gate
    raw_init       : K × f32   (4*K B)   — initial gate params (always 0
                                          for federated training; the swarm
                                          fills these in)

  Total: 32 + 8*K bytes.  For K=5000: ~40 032 B (40 KB).

The TS-side parser lives in src/ntk-gate.ts and MUST match this exactly.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ntkmirror.controller import ForwardFineTuner
from ntkmirror.data import Example


MAGIC = b"NTKG"              # 0x47 0x4B 0x54 0x4E little-endian = 0x4E544B47
HEADER_FMT = "<4sIIIIfQ"     # magic, version, K, n_layers, hidden_size, max_log_gate, model_id_hash
HEADER_SIZE = 32


def fnv1a_64(s: str) -> int:
    """Stable 64-bit hash of the model name; portable to TS for sanity check."""
    h = 0xCBF29CE484222325
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load_examples(path: Path) -> list[Example]:
    """Read a JSONL of {prompt, completion} dicts in the ntkmirror format."""
    out: list[Example] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Example(prompt=obj["prompt"], completion=obj["completion"]))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract top-K NTK-Mirror gates as a postnet-cf binary artifact"
    )
    ap.add_argument("--model", required=True,
                    help="HF model name (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--train", required=True, type=Path,
                    help="JSONL training examples for gate selection")
    ap.add_argument("--gates", type=int, default=5000,
                    help="K = number of gates to select (default 5000)")
    ap.add_argument("--max-log-gate", type=float, default=0.05,
                    help="Bound on |s_{l,c}| (default 0.05)")
    ap.add_argument("--layers", default="all",
                    help="Decoder layer range to score (default 'all')")
    ap.add_argument("--score-batches", type=int, default=16,
                    help="Number of batches used during scoring (default 16)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size during scoring (default 8)")
    ap.add_argument("--max-length", type=int, default=512,
                    help="Tokenizer max_length (default 512)")
    ap.add_argument("--device", default="auto",
                    help="cpu | mps | cuda | auto (default auto)")
    ap.add_argument("--dtype", default="fp32",
                    choices=["fp32", "bf16", "fp16"],
                    help="Model dtype (default fp32 — REQUIRED on MPS to avoid NaN)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output binary path (e.g. public/data/qwen05b-math-gates-k5000.bin)")
    return ap.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"model:  {args.model}")
    print(f"train:  {args.train}")
    print(f"gates:  {args.gates}")
    print(f"out:    {args.out}")
    print()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,   # transformers 5.x uses `dtype` not `torch_dtype`
        device_map={"": device.type},
    )

    examples = load_examples(args.train)
    print(f"loaded {len(examples)} training examples")

    ftt = ForwardFineTuner(
        model,
        tokenizer,
        gates=args.gates,
        layers=args.layers,
        max_log_gate=args.max_log_gate,
    )
    n_layers = len(ftt.decoder_layers)
    hidden_size = ftt.hidden_size
    print(f"model has {n_layers} decoder layers × {hidden_size} hidden = "
          f"{n_layers * hidden_size} candidate gates")
    print(f"selecting top {args.gates} ({100*args.gates/(n_layers*hidden_size):.1f}%) "
          f"by |dL/ds| over {args.score_batches} batches...")

    layer_idx, channel_idx = ftt._score_gates(
        examples,
        score_batches=args.score_batches,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    K = int(layer_idx.numel())
    print(f"selected {K} gates")

    # Sanity check
    li = layer_idx.cpu().to(torch.int64).numpy()
    ci = channel_idx.cpu().to(torch.int64).numpy()
    assert li.shape == ci.shape == (K,)
    assert (0 <= li.min()) and (li.max() < n_layers), \
        f"layer index out of range [0, {n_layers})"
    assert (0 <= ci.min()) and (ci.max() < hidden_size), \
        f"channel index out of range [0, {hidden_size})"
    if hidden_size > 0xFFFF:
        raise ValueError(f"hidden_size={hidden_size} > uint16 max; bump format to uint32")
    if n_layers > 0xFFFF:
        raise ValueError(f"n_layers={n_layers} > uint16 max; bump format to uint32")

    # Per-layer histogram for the writeup
    counts = {int(l): int((li == l).sum()) for l in sorted(set(li.tolist()))}
    print("per-layer gate counts:", counts)

    # Write artifact
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_hash = fnv1a_64(args.model)
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        1,                        # version
        K,
        n_layers,
        hidden_size,
        float(args.max_log_gate),
        model_hash,
    )
    assert len(header) == HEADER_SIZE, f"header is {len(header)} not {HEADER_SIZE}"

    layer_bytes = li.astype("<u2").tobytes()
    channel_bytes = ci.astype("<u2").tobytes()
    raw_bytes = bytes(K * 4)      # K zeros as float32 LE = 4K bytes of \x00

    total = header + layer_bytes + channel_bytes + raw_bytes
    args.out.write_bytes(total)

    print()
    print(f"wrote {args.out}  ({len(total)} bytes, ~{len(total)/1024:.1f} KB)")
    print(f"  header:          {HEADER_SIZE} B")
    print(f"  layer_indices:   {len(layer_bytes)} B")
    print(f"  channel_indices: {len(channel_bytes)} B")
    print(f"  raw_init:        {len(raw_bytes)} B (zeros)")
    print(f"  model_id_hash:   0x{model_hash:016x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
