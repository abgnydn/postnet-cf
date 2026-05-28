#!/usr/bin/env python3
"""
Phase 40 next-4-b session 2 — quantize the Qwen+gates ONNX for browser deployment.

Takes the fp32 ONNX produced by scripts/export-qwen-with-gates.py
(~1.8 GB total via external-data sidecar) and emits a quantized version
small enough to ship to browser tabs.

Two quantization modes:
  int8 (default):  ~500 MB. Easiest. Uses onnxruntime.quantization.quantize_dynamic.
                   Per-channel weight quantization; activations stay fp32 by default.
  int4:            ~250 MB. Needs matmul_4bits_quantizer (gpt-q style block-wise
                   quantization). More tooling-dependent.

Validation: run the quantized ONNX with all-ones gate_mults on a fixed
sample input; compare logits to the fp32 reference. Acceptable tolerance
is generous (~1e-1) because int8/int4 quantization changes per-channel
behavior significantly; what we care about is that the model still
produces plausible logits (no NaN, top-1 token roughly matches).

Run via the ntkmirror venv:
  ~/ntkmirror/.venv/bin/python scripts/quantize-qwen-onnx.py
  ~/ntkmirror/.venv/bin/python scripts/quantize-qwen-onnx.py --mode int4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path("public/data/qwen05b-with-gates.onnx"))
    ap.add_argument("--mode", choices=["int8", "int4"], default="int8")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output ONNX path; defaults to <in_stem>-<mode>.onnx")
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Used only for tokenizer + reference input creation")
    return ap.parse_args()


def quantize_int8(in_path: Path, out_path: Path) -> None:
    """Dynamic int8 quantization via onnxruntime.quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"int8 dynamic quantization: {in_path} → {out_path}")
    t0 = time.time()
    # MatMul + Gather are the size-dominant ops in a transformer; quantizing
    # them gets us most of the size reduction. Per-channel keeps quality.
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")


def quantize_int4(in_path: Path, out_path: Path) -> None:
    """4-bit block-wise weight quantization via matmul_4bits_quantizer.

    Note: this API changes between onnxruntime versions. We try the
    canonical entry point first, then fall back to the legacy one.
    """
    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            MatMul4BitsQuantizer, RTNWeightOnlyQuantConfig,
        )
        config = RTNWeightOnlyQuantConfig(block_size=128)
        print(f"int4 RTN quantization: {in_path} → {out_path}")
        t0 = time.time()
        q = MatMul4BitsQuantizer(
            model=str(in_path),
            algo_config=config,
        )
        q.process()
        q.model.save_model_to_file(str(out_path), use_external_data_format=True)
        print(f"  done in {time.time() - t0:.1f}s")
        return
    except (ImportError, AttributeError) as e:
        print(f"  modern int4 API not available ({e!r}); trying legacy path")

    # Older onnxruntime versions expose a simpler API
    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer  # type: ignore
        print(f"int4 (legacy) quantization: {in_path} → {out_path}")
        t0 = time.time()
        q = MatMul4BitsQuantizer(str(in_path), block_size=128, is_symmetric=True)
        q.process()
        q.model.save_model_to_file(str(out_path), use_external_data_format=True)
        print(f"  done in {time.time() - t0:.1f}s")
    except Exception as e:
        raise RuntimeError(
            f"int4 quantization not supported in this onnxruntime build: {e!r}. "
            "Try `pip install --upgrade onnxruntime` or fall back to --mode int8."
        )


def validate(fp32_path: Path, q_path: Path, model_name: str) -> None:
    """Forward-compare fp32 vs quantized on a fixed sample input.

    With gate_mults = ones, both should produce nearly-equal logits;
    int8 typically diverges by < 0.1 in logits, int4 by < 0.5.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("\nvalidating quantized ONNX against fp32 reference...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    sample = "Question: 14 + 27 = ?\nAnswer:"
    batch = tokenizer([sample], return_tensors="np", padding=True, truncation=True, max_length=32)
    input_ids = batch["input_ids"].astype(np.int64)
    attn_mask = batch["attention_mask"].astype(np.int64)
    gate_mults = np.ones((24, 896), dtype=np.float32)

    def run(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return sess.run(["logits"], {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "gate_mults": gate_mults,
        })[0]

    t0 = time.time()
    logits_fp32 = run(fp32_path)
    fp32_ms = (time.time() - t0) * 1000
    t0 = time.time()
    logits_q = run(q_path)
    q_ms = (time.time() - t0) * 1000

    diff = np.abs(logits_fp32 - logits_q)
    print(f"  forward time:  fp32 {fp32_ms:.0f} ms · quantized {q_ms:.0f} ms")
    print(f"  logits shape:  {logits_q.shape}")
    print(f"  logits diff:   max abs = {diff.max():.4f}, mean abs = {diff.mean():.4f}")

    # Top-1 token agreement (this is the metric that matters for generation)
    top1_fp32 = logits_fp32[0, -1].argmax()
    top1_q = logits_q[0, -1].argmax()
    same_top1 = (top1_fp32 == top1_q)
    print(f"  top-1 next token: fp32={int(top1_fp32)} ('{tokenizer.decode([int(top1_fp32)])}') "
          f"vs quantized={int(top1_q)} ('{tokenizer.decode([int(top1_q)])}')  "
          f"{'✓ match' if same_top1 else '⚠ differ'}")


def main() -> int:
    args = parse_args()
    if not args.in_path.exists():
        print(f"input not found: {args.in_path}")
        print("regenerate via: ~/ntkmirror/.venv/bin/python scripts/export-qwen-with-gates.py")
        return 1

    out_path = args.out or args.in_path.with_name(
        f"{args.in_path.stem}-{args.mode}.onnx"
    )
    print(f"in:    {args.in_path}  ({args.in_path.stat().st_size / 1024 / 1024:.1f} MB)")
    sidecar = args.in_path.with_suffix(args.in_path.suffix + ".data")
    if sidecar.exists():
        print(f"data:  {sidecar}        ({sidecar.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"out:   {out_path}")
    print(f"mode:  {args.mode}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "int8":
        quantize_int8(args.in_path, out_path)
    elif args.mode == "int4":
        quantize_int4(args.in_path, out_path)
    else:
        raise ValueError(args.mode)

    # Report sizes
    if out_path.exists():
        size = out_path.stat().st_size
        sidecar_q = out_path.with_suffix(out_path.suffix + ".data")
        total = size + (sidecar_q.stat().st_size if sidecar_q.exists() else 0)
        print(f"\nout size:  {size / 1024 / 1024:.1f} MB")
        if sidecar_q.exists():
            print(f"  + data:  {sidecar_q.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  total:   {total / 1024 / 1024:.1f} MB")
    else:
        print(f"⚠ {out_path} not created — quantization failed silently?")
        return 1

    if args.validate:
        validate(args.in_path, out_path, args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
