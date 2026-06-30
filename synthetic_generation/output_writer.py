from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from synthetic_generation.config import GenerationConfig
from synthetic_generation.generation_loop import GeneratedUser


logger = logging.getLogger(__name__)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to write synthetic parquet. Run this from the "
            "core_v1 uv environment or install pyarrow."
        ) from exc
    return pa, pq


def _row_from_user(user: GeneratedUser, config: GenerationConfig) -> dict:
    return {
        "synthetic_user_id": user.synthetic_user_id,
        "input_ids": user.input_ids,
        "attention_mask": user.attention_mask,
        "num_tokens": user.num_tokens,
        "num_transactions": user.num_transactions,
        "target_transactions": user.target_transactions,
        "stopped_reason": user.stopped_reason,
        "source": "synthetic_nubank_hyoga",
        "generation_mode": config.mode,
        "teacher_run_id": config.teacher_run_id or "",
        "target_run_id": config.target_run_id or "",
        "teacher_ckpt": config.teacher_ckpt,
        "target_ckpt": config.target_ckpt or "",
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "alpha": float(config.alpha),
        "gamma": float(config.gamma),
        "seed": int(config.seed),
        "classification_loss_mask": 0,
        "ntp_loss_mask": 1,
        # Dummy label for compatibility with CollateWithPromptAndBinaryLabels.
        # Synthetic rows should be used NTP-only unless downstream code honors
        # classification_loss_mask.
        "defaults_6m": 0,
    }


class SyntheticParquetWriter:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.output_dir = config.local_output_dir
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._shard_idx = 0
        self._written = 0

    def write_user(self, user: GeneratedUser) -> None:
        self._buffer.append(_row_from_user(user, self.config))
        if len(self._buffer) >= self.config.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        pa, pq = _require_pyarrow()
        path = self.output_dir / f"part-{self._shard_idx:05d}.parquet"
        table = pa.Table.from_pylist(self._buffer)
        pq.write_table(table, path)
        logger.info("Wrote %d synthetic rows to %s", len(self._buffer), path)
        self._written += len(self._buffer)
        self._buffer.clear()
        self._shard_idx += 1

    def close(self) -> None:
        self.flush()
        config_dict = asdict(self.config)
        config_dict["lengths"] = list(self.config.lengths)
        with open(self.output_dir / "generation_config.json", "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        if self.config.index_litdata:
            try:
                import litdata

                litdata.index_parquet_dataset(str(self.output_dir))
                logger.info("Indexed LitData parquet dataset at %s", self.output_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to index LitData dataset at {self.output_dir}: {exc}") from exc

        if self.config.output_is_gcs:
            _sync_to_gcs(self.output_dir, self.config.output_path)

    @property
    def written(self) -> int:
        return self._written + len(self._buffer)


def write_users(users: Iterable[GeneratedUser], config: GenerationConfig) -> int:
    writer = SyntheticParquetWriter(config)
    try:
        for user in users:
            writer.write_user(user)
        writer.close()
        return writer.written
    finally:
        writer.flush()


def _sync_to_gcs(local_dir: Path, gcs_dir: str) -> None:
    logger.info("Syncing synthetic dataset to %s", gcs_dir)
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_dir.rstrip("/") + "/"], check=True)

