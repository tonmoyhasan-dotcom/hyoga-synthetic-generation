from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    mode: str
    teacher_ckpt: str
    output_path: str
    target_ckpt: str | None = None
    teacher_run_id: str | None = None
    target_run_id: str | None = None
    tokenizer_path: str | None = None
    num_samples: int | None = None
    lengths: tuple[int, ...] = (100,)
    samples_per_length: int | None = None
    max_transactions: int = 2250
    max_context_tokens: int = 4096
    max_tokens_per_transaction: int = 12
    temperature: float = 0.9
    top_p: float = 0.95
    alpha: float = 0.05
    gamma: float = 0.0
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bf16"
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    sep_token_id: int | None = None
    grammar_mode: str = "basic"
    disallow_label_tokens: bool = True
    shard_size: int = 1000
    index_litdata: bool = False
    # Real-prefix seeding: this data has no BOS token (sequences begin mid-transaction), so we
    # bootstrap each synthetic user from the first ``seed_prefix_transactions`` real transactions
    # of a randomly drawn real user, then let G generate the continuation. When seed_source is
    # None and there is no BOS, generation cannot start.
    seed_source: str | None = None
    seed_prefix_transactions: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"vanilla", "contrastive"}:
            raise ValueError(f"mode must be 'vanilla' or 'contrastive', got {self.mode!r}")
        if self.mode == "contrastive" and not self.target_ckpt:
            raise ValueError("contrastive generation requires --target-ckpt")
        if self.num_samples is None and self.samples_per_length is None:
            raise ValueError("provide either --num-samples or --samples-per-length")
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("--num-samples must be >= 1")
        if self.samples_per_length is not None and self.samples_per_length < 1:
            raise ValueError("--samples-per-length must be >= 1")
        if not self.lengths:
            raise ValueError("--lengths must contain at least one transaction length")
        if any(length < 1 for length in self.lengths):
            raise ValueError(f"all lengths must be >= 1, got {self.lengths}")
        if any(length > self.max_transactions for length in self.lengths):
            raise ValueError(f"lengths {self.lengths} exceed max_transactions={self.max_transactions}")
        if self.max_context_tokens < 1:
            raise ValueError("--max-context-tokens must be >= 1")
        if self.max_tokens_per_transaction < 1:
            raise ValueError("--max-tokens-per-transaction must be >= 1")
        if self.temperature <= 0:
            raise ValueError("--temperature must be > 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("--top-p must be in (0, 1]")
        if not (0 < self.alpha <= 1):
            raise ValueError("--alpha must be in (0, 1]")
        if self.gamma < 0:
            raise ValueError("--gamma must be >= 0")
        if self.grammar_mode not in {"basic", "none"}:
            raise ValueError(f"--grammar-mode must be 'basic' or 'none', got {self.grammar_mode!r}")
        if self.shard_size < 1:
            raise ValueError("--shard-size must be >= 1")

    @property
    def output_is_gcs(self) -> bool:
        return self.output_path.startswith("gs://")

    @property
    def local_output_dir(self) -> Path:
        if self.output_is_gcs:
            return Path("/tmp") / "hyoga_synthetic_generation_output"
        return Path(self.output_path)

