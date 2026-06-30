from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from synthetic_generation.config import GenerationConfig
from synthetic_generation.generation_loop import generate_one_user
from synthetic_generation.grammar import BasicTransactionGrammar
from synthetic_generation.length_schedule import build_length_schedule, iter_with_index, parse_lengths
from synthetic_generation.model_loader import load_lightning_generator_model
from synthetic_generation.output_writer import SyntheticParquetWriter
from synthetic_generation.seed_prefixes import load_seed_prefixes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic HFv3/Hyoga transaction histories.")
    parser.add_argument("--mode", choices=["vanilla", "contrastive"], required=True)
    parser.add_argument("--teacher-ckpt", required=True, help="GCS/local Lightning checkpoint for stronger generator G")
    parser.add_argument("--target-ckpt", default=None, help="GCS/local Lightning checkpoint for frozen reference M0")
    parser.add_argument("--teacher-run-id", default=None)
    parser.add_argument("--target-run-id", default=None)
    parser.add_argument("--tokenizer-path", default=None, help="Reserved for future explicit tokenizer override")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--lengths", default="100", help="Comma-separated target transaction counts")
    parser.add_argument("--samples-per-length", type=int, default=None)
    parser.add_argument("--max-transactions", type=int, default=2250)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens-per-transaction", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--bos-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--sep-token-id", type=int, default=None)
    parser.add_argument("--allow-label-tokens", action="store_true")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--index-litdata", action="store_true")
    parser.add_argument("--seed-source", default=None,
                        help="GCS glob of real tokenized parquet to seed each user's first "
                             "transaction(s) from (required when the model has no BOS token)")
    parser.add_argument("--seed-prefix-transactions", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        mode=args.mode,
        teacher_ckpt=args.teacher_ckpt,
        target_ckpt=args.target_ckpt,
        teacher_run_id=args.teacher_run_id,
        target_run_id=args.target_run_id,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        lengths=parse_lengths(args.lengths),
        samples_per_length=args.samples_per_length,
        max_transactions=args.max_transactions,
        max_context_tokens=args.max_context_tokens,
        max_tokens_per_transaction=args.max_tokens_per_transaction,
        temperature=args.temperature,
        top_p=args.top_p,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        sep_token_id=args.sep_token_id,
        disallow_label_tokens=not args.allow_label_tokens,
        shard_size=args.shard_size,
        index_litdata=args.index_litdata,
        seed_source=args.seed_source,
        seed_prefix_transactions=args.seed_prefix_transactions,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("synthetic_generation")
    config = _build_config(args)

    if config.tokenizer_path:
        logger.warning("--tokenizer-path is recorded in metadata but checkpoint config tokenizer is used in V0")

    schedule = build_length_schedule(
        lengths=config.lengths,
        num_samples=config.num_samples,
        samples_per_length=config.samples_per_length,
        seed=config.seed,
    )
    logger.info("Generation plan: %d samples across lengths=%s", len(schedule), sorted(set(schedule)))

    teacher_loaded = load_lightning_generator_model(
        ckpt_path=config.teacher_ckpt,
        context_length=config.max_context_tokens,
        device=config.device,
    )
    target_model = None
    if config.mode == "contrastive":
        target_loaded = load_lightning_generator_model(
            ckpt_path=config.target_ckpt or "",
            context_length=config.max_context_tokens,
            device=config.device,
        )
        if len(target_loaded.tokenizer) != len(teacher_loaded.tokenizer):
            raise RuntimeError(
                "Teacher and target tokenizers differ in size: "
                f"{len(teacher_loaded.tokenizer)} vs {len(target_loaded.tokenizer)}"
            )
        target_model = target_loaded.model

    grammar = BasicTransactionGrammar.from_tokenizer(
        teacher_loaded.tokenizer,
        vocab_size=int(teacher_loaded.metadata.get("vocab_size") or len(teacher_loaded.tokenizer)),
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        sep_token_id=config.sep_token_id,
        disallow_label_tokens=config.disallow_label_tokens,
    )
    logger.info(
        "Grammar: vocab=%d bos=%s sep=%s eos=%s disallowed=%d",
        grammar.vocab_size,
        grammar.bos_token_id,
        grammar.sep_token_id,
        grammar.eos_token_id,
        len(grammar.disallowed_token_ids),
    )

    device = next(teacher_loaded.model.parameters()).device
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed)

    # Real-prefix seeds (this data has no BOS). One seed per scheduled sample.
    seeds: list[list[int]] | None = None
    if config.seed_source:
        seeds = load_seed_prefixes(
            seed_source_glob=config.seed_source,
            num_prefixes=len(schedule),
            prefix_transactions=config.seed_prefix_transactions,
            sep_token_id=grammar.sep_token_id,
            seed=config.seed,
        )
        logger.info("Loaded %d real seed prefixes (%d-transaction) for generation",
                    len(seeds), config.seed_prefix_transactions)

    writer = SyntheticParquetWriter(config)
    try:
        for sample_idx, target_transactions in iter_with_index(schedule):
            user = generate_one_user(
                config=config,
                teacher=teacher_loaded.model,
                target=target_model,
                grammar=grammar,
                target_transactions=target_transactions,
                sample_idx=sample_idx,
                torch_generator=torch_generator,
                initial_sequence=(seeds[sample_idx] if seeds else None),
            )
            writer.write_user(user)
            if (sample_idx + 1) % 10 == 0:
                logger.info("Generated %d/%d samples", sample_idx + 1, len(schedule))
        writer.close()
    finally:
        writer.flush()

    logger.info("Wrote %d synthetic rows to %s", writer.written, config.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

