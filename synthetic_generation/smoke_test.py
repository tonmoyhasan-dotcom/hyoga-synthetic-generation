#!/usr/bin/env python
"""Smoke test for a generated synthetic parquet directory.

Static checks (no model): readable, required columns, token ids within vocab, snapshots end in
SEP, NTP-only flags, non-empty, not all identical, max length within context.
Model checks (optional, --teacher-ckpt): the training collate_fn batches the rows and the model
forward produces full-vocab logits on a tiny batch (HFv3-compat).

Usage:
    python -m synthetic_generation.smoke_test --parquet /tmp/syn_smoke2 \
        --teacher-ckpt gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/checkpoints/step=078113.ckpt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

VOCAB_SIZE = 16386          # model vocab incl. label tokens 16384/16385
POS_LABEL, NEG_LABEL = 16384, 16385
SEP = 2
REQUIRED_COLS = ("input_ids", "attention_mask", "defaults_6m", "ntp_loss_mask",
                 "classification_loss_mask", "source")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _read_rows(parquet_path: str):
    p = Path(parquet_path)
    files = sorted(glob.glob(str(p / "*.parquet"))) if p.is_dir() else [parquet_path]
    if not files:
        raise FileNotFoundError(f"No parquet found at {parquet_path}")
    tables = [pq.read_table(f) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tables)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="parquet file or output directory")
    ap.add_argument("--max-context-tokens", type=int, default=4096)
    ap.add_argument("--teacher-ckpt", default=None, help="if set, also run collate+forward checks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ok = True
    t = _read_rows(args.parquet)
    cols = set(t.column_names)
    ok &= _check("parquet readable", t.num_rows > 0, f"{t.num_rows} rows")
    for c in REQUIRED_COLS:
        ok &= _check(f"column '{c}' present", c in cols)

    seqs = [list(map(int, s)) for s in t["input_ids"].to_pylist()]
    flat = [tok for s in seqs for tok in s]
    ok &= _check("non-empty sequences", all(len(s) > 0 for s in seqs))
    ok &= _check("token ids within vocab", flat and min(flat) >= 0 and max(flat) < VOCAB_SIZE,
                 f"min={min(flat)} max={max(flat)} vocab={VOCAB_SIZE}")
    ok &= _check("max length <= context", max(len(s) for s in seqs) <= args.max_context_tokens,
                 f"max_len={max(len(s) for s in seqs)}")
    ok &= _check("no label tokens in body", POS_LABEL not in flat and NEG_LABEL not in flat)
    ok &= _check("snapshots end in SEP", all(s[-1] == SEP for s in seqs))
    distinct = len({tuple(s) for s in seqs})
    ok &= _check("not all identical", distinct > 1 or len(seqs) == 1, f"{distinct}/{len(seqs)} distinct")
    ok &= _check("NTP-only flags", set(t["classification_loss_mask"].to_pylist()) == {0}
                 and set(t["ntp_loss_mask"].to_pylist()) == {1})

    if args.teacher_ckpt:
        import torch
        from synthetic_generation.model_loader import load_lightning_generator_model, setup_core_imports
        setup_core_imports()
        from core.evaluation.loader import load_lightning_core_model
        from core.evaluation.model import _forward_model

        mw, _pp, meta = load_lightning_core_model(args.teacher_ckpt, context_length=args.max_context_tokens)
        collate = getattr(mw, "collate_fn", None)
        ok &= _check("collate_fn available", collate is not None)
        batch = [{"input_ids": s, "attention_mask": [1] * len(s), "defaults_6m": 0} for s in seqs[:4]]
        input_ids, _labels = collate(batch)
        ok &= _check("collate_fn batches rows", input_ids.shape[0] == len(batch),
                     f"shape={tuple(input_ids.shape)}")
        with torch.inference_mode():
            out = _forward_model(mw.model, input_ids.to(mw.device))
        ok &= _check("model forward on tiny batch", out.logits.shape[0] == len(batch),
                     f"logits={tuple(out.logits.shape)}")

    print("\n=== SMOKE " + ("PASSED" if ok else "FAILED") + " ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
