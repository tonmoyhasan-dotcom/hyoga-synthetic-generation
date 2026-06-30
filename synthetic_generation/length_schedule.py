from __future__ import annotations

import random
from collections.abc import Iterator


def parse_lengths(raw: str) -> tuple[int, ...]:
    lengths = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not lengths:
        raise ValueError("--lengths produced no values")
    return lengths


def build_length_schedule(
    *,
    lengths: tuple[int, ...],
    num_samples: int | None,
    samples_per_length: int | None,
    seed: int,
) -> list[int]:
    """Build the per-sample target transaction-count schedule.

    If ``samples_per_length`` is set, each requested length is repeated exactly that
    many times. Otherwise ``num_samples`` is spread as evenly as possible across
    the lengths and shuffled deterministically.
    """
    if samples_per_length is not None:
        schedule = [length for length in lengths for _ in range(samples_per_length)]
    else:
        assert num_samples is not None
        schedule = [lengths[i % len(lengths)] for i in range(num_samples)]

    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def iter_with_index(schedule: list[int]) -> Iterator[tuple[int, int]]:
    for sample_idx, target_transactions in enumerate(schedule):
        yield sample_idx, target_transactions

