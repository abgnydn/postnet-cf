#!/usr/bin/env python3
"""
Phase 40 next-4-b — export Qwen-0.5B-Instruct to ONNX with per-layer
gate-multiplier INPUTS.

The standard Hugging Face ONNX export of a causal LM has signature
  inputs:  input_ids, attention_mask
  outputs: logits
and the model behaves as a frozen function. For postnet-cf's federated
gate-controller training we need each forward to apply DIFFERENT gate
values (the SPSA trial under test). The cleanest way: add a third
input `gate_mults` of shape [n_layers, hidden_size] that gets multiplied
into every decoder layer's residual output. With all-ones the model is
the unchanged base; with non-trivial gate_mults the residual stream is
gated as in NTK-Mirror.

The exported ONNX is a STATIC artifact: ship it once, browsers fetch
it and call session.run() with their per-trial gate_mults vector.

Output:
  public/data/qwen05b-with-gates.onnx  (+ optional .onnx_data sidecar
                                         if the model is large enough
                                         to trigger ONNX external data)

Validation:
  We run the exported ONNX with all-ones gate_mults and compare logits
  to the unwrapped PyTorch model — they must agree within fp32 noise.

Run via the ntkmirror venv (installed in Phase 40 scope):
  ~/ntkmirror/.venv/bin/python scripts/export-qwen-with-gates.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── wrapper: gate_mults applied via forward hooks on decoder layers ─

class QwenWithGateInputs(nn.Module):
    """Frozen Qwen-0.5B-Instruct plus per-layer multiplicative gates passed
    in as a forward input. Hooks fire during torch.onnx.export tracing so
    the multiplications appear as Mul ops in the resulting graph.
    """

    def __init__(self, base_model, n_layers: int, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # gate_mults is stashed on the module during forward() and read
        # by the hooks. torch.onnx.export with a single example input
        # traces the hook bodies, so the Mul ops end up on the graph
        # with gate_mults as their second operand — exactly what we want.
        self._gate_mults: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for i, layer in enumerate(self.base.model.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self._gate_mults is None:
                return output
            mult = self._gate_mults[layer_idx]  # shape [hidden_size]
            shape = (1, 1, -1)
            if isinstance(output, tuple):
                h = output[0]
                gated = h * mult.view(*shape).to(dtype=h.dtype, device=h.device)
                return (gated,) + output[1:]
            return output * mult.view(*shape).to(dtype=output.dtype, device=output.device)
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def forward(self, input_ids, attention_mask, gate_mults):
        # gate_mults: [n_layers, hidden_size] (float32)
        # We stash on self for the hook callbacks. Safe because forward() is
        # synchronous and we clear on exit.
        self._gate_mults = gate_mults
        try:
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            return out.logits
        finally:
            self._gate_mults = None


# ─── export ─

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="public/data/qwen05b-with-gates.onnx",
                    type=Path,
                    help="Output ONNX path (will create parent dirs)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (17 has all we need; 14+ also works)")
    ap.add_argument("--seq-len", type=int, default=32,
                    help="Example seq_len for export tracing (shape becomes dynamic anyway)")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="Run the exported ONNX with all-ones gate_mults and check parity")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    print(f"model: {args.model}")
    print(f"out:   {args.out}")
    print()

    print("loading PyTorch model (fp32 on CPU — ONNX export needs CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    cfg = base_model.config
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    print(f"architecture: {n_layers} layers × {hidden_size} hidden")

    wrapper = QwenWithGateInputs(base_model, n_layers, hidden_size)
    wrapper.eval()

    # Example inputs for tracing
    sample_text = "Question: 14 + 27 = ?\nAnswer:"
    batch = tokenizer([sample_text], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.seq_len)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    gate_mults_ones = torch.ones(n_layers, hidden_size, dtype=torch.float32)
    print(f"example input shapes: ids {tuple(input_ids.shape)}, "
          f"mask {tuple(attention_mask.shape)}, gate_mults {tuple(gate_mults_ones.shape)}")

    # Sanity: PyTorch forward with all-ones gates == unwrapped model
    with torch.no_grad():
        unwrapped = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        wrapped_ones = wrapper(input_ids, attention_mask, gate_mults_ones)
        torch_diff = (wrapped_ones - unwrapped).abs().max().item()
    print(f"PyTorch wrap-with-ones diff vs unwrapped: max abs = {torch_diff:.2e}")
    assert torch_diff < 1e-5, "wrapper with all-ones != base; hook math is wrong"
    print("  ✓ wrapper math is a no-op at gate_mults = ones")

    # Export
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nexporting to {args.out}  (opset {args.opset})...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, gate_mults_ones),
        str(args.out),
        input_names=["input_ids", "attention_mask", "gate_mults"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            # gate_mults stays static [n_layers, hidden_size]
        },
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
    )
    wrapper.remove_hooks()
    elapsed = time.time() - t0
    size_bytes = args.out.stat().st_size
    print(f"  done in {elapsed:.1f}s · {size_bytes} bytes "
          f"({size_bytes / 1024 / 1024:.1f} MB)")

    # Also check if external-data sidecars were emitted (Qwen-0.5B fp32 weights
    # are ~1 GB which is larger than the 2 GB protobuf limit on most setups
    # but still might trigger external data)
    sidecars = list(args.out.parent.glob(f"{args.out.stem}*.onnx_data")) + \
               list(args.out.parent.glob(f"{args.out.name}_data")) + \
               list(args.out.parent.glob(f"{args.out.stem}.data"))
    if sidecars:
        for s in sidecars:
            print(f"  + sidecar: {s} ({s.stat().st_size} bytes)")

    # Validation: onnxruntime forward with all-ones gates ≈ unwrapped PyTorch
    if args.validate:
        try:
            import onnxruntime as ort
        except ImportError:
            print("\nonnxruntime not installed; skipping ONNX validation")
            print("(install via: ~/ntkmirror/.venv/bin/pip install onnxruntime)")
        else:
            print("\nvalidating exported ONNX...")
            sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
            outs = sess.run(
                ["logits"],
                {
                    "input_ids": input_ids.numpy(),
                    "attention_mask": attention_mask.numpy(),
                    "gate_mults": gate_mults_ones.numpy(),
                },
            )
            onnx_logits = outs[0]
            ort_diff = float(np.abs(onnx_logits - unwrapped.numpy()).max())
            print(f"  ONNX-vs-PyTorch logits max abs diff: {ort_diff:.2e}")
            if ort_diff > 1e-3:
                print("  ⚠ diff is larger than 1e-3 — verify before shipping")
                return 1
            else:
                print("  ✓ ONNX export matches PyTorch within fp32 noise")

    return 0


if __name__ == "__main__":
    sys.exit(main())
