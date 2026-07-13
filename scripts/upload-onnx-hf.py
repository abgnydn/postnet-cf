#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""
Phase 40 next-7 — publish the Qwen-0.5B + NTK-Mirror-gates int8 ONNX to
HuggingFace Hub so the /ntk.html demo works for a stranger with no local
server (the deployed worker fetches the model straight from HF's CDN).

Self-contained: run with uv, no venv needed. Uses your cached HF token
(`~/.cache/huggingface/token`, from `huggingface-cli login` / `hf auth login`)
or the HF_TOKEN env var.

    uv run scripts/upload-onnx-hf.py \
        --file ~/postnet-cf-onnx/qwen05b-with-gates-optimum-int8.onnx

Defaults publish to:
    https://huggingface.co/abgunaydin/postnet-qwen05b-with-gates
        /resolve/main/qwen05b-with-gates-optimum-int8.onnx

which is exactly the HF_ONNX_URL default in public/ntk-worker.js. If you
change --repo or --path-in-repo here, update that constant to match.

The ~906 MB file is an int8 derivative of Qwen2.5-0.5B-Instruct (Apache-2.0)
with 24 forward-only gate `Mul` nodes injected (NTK-Mirror parameterization,
MIT). Redistributing the derivative is permitted under Apache-2.0; the model
card uploaded alongside it carries the attribution.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

DEFAULT_FILE = Path.home() / "postnet-cf-onnx" / "qwen05b-with-gates-optimum-int8.onnx"
DEFAULT_REPO = "abgunaydin/postnet-qwen05b-with-gates"
DEFAULT_PATH_IN_REPO = "qwen05b-with-gates-optimum-int8.onnx"

MODEL_CARD = """\
---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: onnxruntime
tags:
  - onnx
  - int8
  - federated-learning
  - ntk-mirror
  - postnet-cf
---

# Qwen2.5-0.5B-Instruct + NTK-Mirror gates (int8 ONNX)

Single-file int8 ONNX (~906 MB) used by the browser workers in
[**postnet-cf**](https://github.com/abgnydn/postnet-cf) — federated learning
where the workers are browser tabs and the coordinator is a Cloudflare
Durable Object. The `/ntk.html` demo runs this model forward *in the tab* via
`onnxruntime-web` and federated-SPSA-trains a sparse NTK-Mirror gate
controller (K = 5000 signed log-gates) across everyone who opens the URL.

**▶ Live demo:** <https://postnet-cf.abgunaydin94.workers.dev/ntk> — open it,
click Join, and your tab starts federated-SPSA-training against everyone else
who's connected. No install, no clone; this model streams in from the Hub.

## What this is

- **Base:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  (Apache-2.0), exported with `optimum-cli` (ORT-web compatible), then
  int8-dynamic-quantized to a single file (no external-data sidecar) so it
  fits in one `ArrayBuffer` and loads in Chrome's `onnxruntime-web` 1.22.
- **Gate surgery:** 24 forward-only `Mul` nodes injected on each decoder
  layer's post-MLP residual, driven by an extra graph input
  `gate_mults: [24, 896] float32`. With `gate_mults = 1` the graph is
  byte-identical to the base export. Parameterization from
  [NTK-Mirror](https://github.com/leochlon/ntkmirror) (MIT).

## Inputs / outputs

| input | dtype | shape |
|---|---|---|
| `input_ids` | int64 | `[batch, seq]` |
| `attention_mask` | int64 | `[batch, seq]` |
| `position_ids` | int64 | `[batch, seq]` |
| `gate_mults` | float32 | `[24, 896]` |

Output: `logits` `[batch, seq, vocab]`.

## License / attribution

Apache-2.0 (inherits the base model's license). This is a **quantized
derivative** of Qwen2.5-0.5B-Instruct © the Alibaba Qwen team. The gate
parameterization is from Leon Chlon's NTK-Mirror (Hassana Labs), MIT.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload the gated Qwen int8 ONNX to HF Hub.")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE,
                    help=f"Path to the int8 ONNX (default: {DEFAULT_FILE})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF model repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO,
                    help=f"Destination filename in the repo (default: {DEFAULT_PATH_IN_REPO})")
    ap.add_argument("--private", action="store_true",
                    help="Create the repo private (default: public, so the demo can fetch it)")
    ap.add_argument("--no-card", action="store_true",
                    help="Skip uploading/refreshing the model card")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check token + file + repo, print the plan, upload nothing")
    return ap.parse_args()


def die(msg: str, code: int = 1) -> "None":
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    args = parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.utils import get_token

    token = get_token()
    if not token:
        die("no HuggingFace token found. Log in first:\n"
            "    uv run --with huggingface_hub python -c \"from huggingface_hub import login; login()\"\n"
            "  or set HF_TOKEN, or run `huggingface-cli login`.")

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as e:  # noqa: BLE001
        die(f"token present but whoami() failed ({e}). Token may be expired or read-only.")
    print(f"authenticated as: {user}")

    f = args.file.expanduser()
    if not f.is_file():
        die(f"ONNX file not found: {f}\n\n"
            "  This machine may not have the built artifact. Rebuild it with the\n"
            "  Phase 40-4b-s5 pipeline (needs ~/ntkmirror venv + Qwen + optimum):\n"
            "    optimum-cli export onnx -m Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "        --task text-generation --opset 17 --monolith ~/postnet-cf-onnx/qwen05b-optimum/\n"
            "    python scripts/inject-gates-onnx.py \\\n"
            "        --out ~/postnet-cf-onnx/qwen05b-with-gates-optimum.onnx\n"
            "    python scripts/quantize-qwen-onnx.py \\\n"
            "        --in  ~/postnet-cf-onnx/qwen05b-with-gates-optimum.onnx \\\n"
            "        --out ~/postnet-cf-onnx/qwen05b-with-gates-optimum-int8.onnx\n"
            "  ...then re-run this script (or copy the file to this machine and pass --file).")

    size_mb = f.stat().st_size / 1024 / 1024
    resolve_url = f"https://huggingface.co/{args.repo}/resolve/main/{args.path_in_repo}"
    print(f"file:        {f}  ({size_mb:.0f} MB)")
    print(f"→ repo:      https://huggingface.co/{args.repo}  ({'private' if args.private else 'public'})")
    print(f"→ path:      {args.path_in_repo}")
    print(f"→ resolve:   {resolve_url}")

    if args.dry_run:
        print("\n[dry-run] token OK, file present. Nothing uploaded.")
        return 0

    print(f"\ncreating repo {args.repo} (exist_ok)...")
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    if not args.no_card:
        print("uploading model card (README.md)...")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
            tf.write(MODEL_CARD)
            card_path = tf.name
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model",
                        commit_message="Model card: attribution + I/O signature")

    print(f"uploading {size_mb:.0f} MB ONNX (resumable; may take a while)...")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=args.path_in_repo,
                    repo_id=args.repo, repo_type="model",
                    commit_message="Publish Qwen-0.5B + NTK-Mirror gates int8 ONNX for postnet-cf /ntk.html")

    print("\n✓ upload complete.")
    print(f"  Public URL: {resolve_url}")
    print("  This matches HF_ONNX_URL in public/ntk-worker.js — the deployed")
    print("  worker will now load the model with no local server. Verify with:")
    print(f"    curl -sIL '{resolve_url}' | grep -i -E 'HTTP/|content-length|access-control-allow-origin'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
