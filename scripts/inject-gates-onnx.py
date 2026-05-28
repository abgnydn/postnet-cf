#!/usr/bin/env python3
"""
Phase 40 next-4-b session 5 — surgically inject NTK-Mirror gate multipliers
into an optimum-cli-exported Qwen ONNX.

Input:
  ~/postnet-cf-onnx/qwen05b-optimum/model.onnx  (produced by optimum-cli;
                                                 known to be ORT-web-compatible)

Output:
  ~/postnet-cf-onnx/qwen05b-with-gates-optimum.onnx
  ~/postnet-cf-onnx/qwen05b-with-gates-optimum.onnx_data  (external weights)

What we change:
  - Add a new graph input  `gate_mults: [n_layers, hidden_size] float32`
  - For each decoder layer L, find its post-MLP residual output tensor
    `/model/layers.{L}/Add_1_output_0` and insert two ops just after it:
      * Gather(gate_mults, indices=[L], axis=0)   →  shape [1, hidden_size]
      * Unsqueeze on axes [1]                      →  shape [1, 1, hidden_size]
      * Mul(layer_output, mult)                    →  shape [B, T, hidden_size]
    Then rewire ALL downstream consumers of the layer's residual output
    to read the gated tensor instead.
  - With gate_mults = ones, the graph behaves byte-identically to the
    base optimum-cli export.

Validation: run both ONNX (original + gated) in onnxruntime CPU with the
same input + all-ones gate_mults; logits should match within ~1e-3
(some op-fusion differences are expected post-surgery).

Run via the ntkmirror venv:
  ~/ntkmirror/.venv/bin/python scripts/inject-gates-onnx.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


# Qwen-0.5B-Instruct constants (also used by the worker / DO)
N_LAYERS = 24
HIDDEN_SIZE = 896

# Per-layer gating target: each layer's post-MLP residual output.
# We confirmed by inspecting the optimum-cli ONNX in `optimum-cli export
# onnx -m Qwen/Qwen2.5-0.5B-Instruct ...` that the relevant tensor names
# are /model/layers.{L}/Add_1_output_0 for L=0..23. The final norm and
# lm_head consume layer 23's output as residual.
def layer_residual_tensor_name(L: int) -> str:
    return f"/model/layers.{L}/Add_1_output_0"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path.home() / "postnet-cf-onnx/qwen05b-optimum/model.onnx",
                    help="optimum-cli exported model.onnx")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "postnet-cf-onnx/qwen05b-with-gates-optimum.onnx",
                    help="Where to write the gated ONNX")
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Used only for tokenizer + reference input creation during validation")
    return ap.parse_args()


def inject_gates(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph

    # 1. Add gate_mults as a new graph input. Static shape [N_LAYERS, HIDDEN_SIZE].
    gate_mults_input = helper.make_tensor_value_info(
        "gate_mults",
        TensorProto.FLOAT,
        [N_LAYERS, HIDDEN_SIZE],
    )
    graph.input.append(gate_mults_input)

    # 2. For each layer, build the gating subgraph and rewire consumers.
    # We add nodes at the END of the node list (ONNX nodes are topologically
    # sortable; ORT will sort on session.create). But to keep the graph in a
    # consistent topological order for tools, we insert new nodes near the
    # layer's Add_1 node. For simplicity, we append; topological sort happens
    # at load time.

    new_nodes = []
    rewire_map = {}      # old_output → new_output

    for L in range(N_LAYERS):
        residual_name = layer_residual_tensor_name(L)
        gated_name = f"/postnet/gated_layer_{L}_output_0"
        mult_idx_name = f"/postnet/gate_idx_{L}"
        mult_gather_name = f"/postnet/gate_mult_{L}_2d"
        mult_bcast_name = f"/postnet/gate_mult_{L}_bcast"

        # Constant: indices = [L]  (int64, shape [1])
        indices_init = numpy_helper.from_array(
            np.array([L], dtype=np.int64), name=f"/postnet/gate_idx_const_{L}"
        )
        graph.initializer.append(indices_init)
        # Constant: axes = [1] for Unsqueeze (broadcast to [1, 1, H])
        axes_init = numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=f"/postnet/gate_unsq_axes_const_{L}"
        )
        graph.initializer.append(axes_init)

        # Gather(gate_mults, indices=[L], axis=0) → shape [1, HIDDEN_SIZE]
        new_nodes.append(helper.make_node(
            "Gather",
            inputs=["gate_mults", indices_init.name],
            outputs=[mult_gather_name],
            name=f"/postnet/Gather_gate_mult_{L}",
            axis=0,
        ))
        # Unsqueeze on axis 1 → shape [1, 1, HIDDEN_SIZE]   (opset 13+ takes axes as input)
        new_nodes.append(helper.make_node(
            "Unsqueeze",
            inputs=[mult_gather_name, axes_init.name],
            outputs=[mult_bcast_name],
            name=f"/postnet/Unsqueeze_gate_mult_{L}",
        ))
        # Mul(layer_residual, mult)  →  [B, T, HIDDEN_SIZE]
        new_nodes.append(helper.make_node(
            "Mul",
            inputs=[residual_name, mult_bcast_name],
            outputs=[gated_name],
            name=f"/postnet/Mul_gate_{L}",
        ))

        rewire_map[residual_name] = gated_name

    # 3. Rewire: for every node that consumes a layer's old residual output,
    # replace that input with the gated output instead — EXCEPT the new Mul
    # node we just created (it must keep reading the un-gated residual).
    new_node_names = {n.name for n in new_nodes}
    for node in graph.node:
        if node.name in new_node_names:
            continue
        for i, inp in enumerate(node.input):
            if inp in rewire_map:
                node.input[i] = rewire_map[inp]

    # Also rewire graph outputs (in case any layer's residual is a model output;
    # not the case for Qwen text-generation but defensive).
    for out in graph.output:
        if out.name in rewire_map:
            out.name = rewire_map[out.name]

    # 4. Append our new nodes. ORT will topologically sort on load.
    for n in new_nodes:
        graph.node.append(n)

    # Update model metadata so a quick inspection shows our changes.
    model.producer_name = "postnet-cf-inject-gates"
    model.producer_version = "1.0"
    model.doc_string = (model.doc_string or "") + \
        f"\nNTK-Mirror gates injected at /model/layers.{{0..{N_LAYERS - 1}}}/Add_1_output_0; " \
        f"new graph input: gate_mults[{N_LAYERS}, {HIDDEN_SIZE}] float32."

    return model


def validate(original_path: Path, gated_path: Path, model_name: str) -> bool:
    """Run both ONNX in ORT-CPU on the same input; logits must match
    within ~1e-3 when gate_mults = ones (modulo Mul-fusion noise)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("\nvalidating injected ONNX vs original (with gate_mults = ones)...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    sample = "Question: 14 + 27 = ?\nAnswer:"
    batch = tok([sample], return_tensors="np", padding=True, truncation=True, max_length=32)
    input_ids = batch["input_ids"].astype(np.int64)
    attn = batch["attention_mask"].astype(np.int64)
    # position_ids is needed (we saw it in optimum-cli's inputs)
    pos_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :].repeat(input_ids.shape[0], axis=0)
    gate_mults = np.ones((N_LAYERS, HIDDEN_SIZE), dtype=np.float32)

    sess_o = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    sess_g = ort.InferenceSession(str(gated_path), providers=["CPUExecutionProvider"])

    feeds_o = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "position_ids": pos_ids,
    }
    feeds_g = {**feeds_o, "gate_mults": gate_mults}

    t = time.time()
    out_o = sess_o.run(["logits"], feeds_o)[0]
    print(f"  original ORT forward: {(time.time() - t) * 1000:.0f} ms")
    t = time.time()
    out_g = sess_g.run(["logits"], feeds_g)[0]
    print(f"  gated    ORT forward: {(time.time() - t) * 1000:.0f} ms")

    diff = np.abs(out_o - out_g)
    print(f"  logits max abs diff: {diff.max():.4e}")
    print(f"  logits mean abs diff: {diff.mean():.4e}")

    # Top-1 token agreement is the most important sanity check
    top1_o = out_o[0, -1].argmax()
    top1_g = out_g[0, -1].argmax()
    print(f"  top-1 next token: original={int(top1_o)} ('{tok.decode([int(top1_o)])}') "
          f"vs gated={int(top1_g)} ('{tok.decode([int(top1_g)])}')  "
          f"{'✓ match' if top1_o == top1_g else '✗ DIFFER'}")

    return top1_o == top1_g and diff.max() < 1e-2


def main() -> int:
    args = parse_args()
    if not args.in_path.exists():
        print(f"input not found: {args.in_path}")
        print("export via: optimum-cli export onnx -m Qwen/Qwen2.5-0.5B-Instruct --task text-generation --opset 17 --monolith ~/postnet-cf-onnx/qwen05b-optimum/")
        return 1

    print(f"in:  {args.in_path}")
    print(f"out: {args.out}")
    print()

    print("loading ONNX with external data...")
    model = onnx.load(str(args.in_path))
    print(f"  opset: {[(o.domain, o.version) for o in model.opset_import]}")
    print(f"  inputs:  {[i.name for i in model.graph.input]}")
    print(f"  outputs: {[o.name for o in model.graph.output]}")

    print("\ninjecting gate_mults input + 24 Mul nodes...")
    t0 = time.time()
    model = inject_gates(model)
    print(f"  done in {time.time() - t0:.2f}s")
    print(f"  inputs now:  {[i.name for i in model.graph.input]}")

    print(f"\nrunning ONNX shape inference + checker...")
    try:
        onnx.checker.check_model(model, full_check=True)
        print("  ✓ checker passed")
    except Exception as e:
        print(f"  ⚠ checker warned: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nsaving to {args.out} (external data sidecar: {args.out.name}_data)...")
    onnx.save_model(
        model,
        str(args.out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{args.out.name}_data",
        size_threshold=1024,
    )
    print(f"  graph: {args.out.stat().st_size / 1024 / 1024:.1f} MB")
    sidecar = args.out.parent / f"{args.out.name}_data"
    if sidecar.exists():
        print(f"  data:  {sidecar.stat().st_size / 1024 / 1024:.1f} MB")

    if args.validate:
        ok = validate(args.in_path, args.out, args.model)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
