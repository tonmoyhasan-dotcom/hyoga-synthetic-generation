"""Real-prefix seeding for autoregressive generation.

This tokenized data has no BOS token (a sequence just begins with the first field of its first
transaction), so generation must be bootstrapped from a real prefix. We take the first
``prefix_transactions`` transactions (up to and including the k-th SEP) of randomly drawn real
users; G then generates the continuation. The short real prefix is recorded as part of the
synthetic row and conditions the model the same way training sequences begin.
"""
from __future__ import annotations

import logging
import random
from typing import List

logger = logging.getLogger(__name__)


def _first_k_transactions(token_ids: List[int], k: int, sep_token_id: int) -> List[int] | None:
    """Return tokens up to and including the k-th SEP, or None if fewer than k transactions."""
    seen = 0
    for i, tok in enumerate(token_ids):
        if int(tok) == sep_token_id:
            seen += 1
            if seen == k:
                return [int(t) for t in token_ids[: i + 1]]
    return None


def load_seed_prefixes(
    *,
    seed_source_glob: str,
    num_prefixes: int,
    prefix_transactions: int,
    sep_token_id: int,
    seed: int,
) -> List[List[int]]:
    """Load ``num_prefixes`` real prefixes (each = first ``prefix_transactions`` transactions).

    Prefixes are sampled (with replacement if the source is smaller than requested) so the caller
    always gets exactly ``num_prefixes`` seeds.
    """
    import gcsfs
    import pyarrow.parquet as pq

    fs = gcsfs.GCSFileSystem()
    files = sorted(fs.glob(seed_source_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files under seed source: {seed_source_glob}")
    rng = random.Random(seed)
    rng.shuffle(files)

    pool: List[List[int]] = []
    for f in files:
        tbl = pq.read_table(fs.open(f), columns=["input_ids"])
        for ids in tbl["input_ids"].to_pylist():
            pref = _first_k_transactions(ids, prefix_transactions, sep_token_id)
            if pref:
                pool.append(pref)
        if len(pool) >= max(num_prefixes, 1000):  # enough variety to sample from
            break
    if not pool:
        raise RuntimeError(
            f"No real sequence with >= {prefix_transactions} transactions (SEP={sep_token_id}) "
            f"found in {seed_source_glob}"
        )
    logger.info("Seed pool: %d real prefixes (%d-transaction) from %d files",
                len(pool), prefix_transactions, len(files))
    return [pool[rng.randrange(len(pool))] for _ in range(num_prefixes)]
