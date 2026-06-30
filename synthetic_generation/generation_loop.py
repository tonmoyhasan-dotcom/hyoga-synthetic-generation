from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from synthetic_generation.config import GenerationConfig
from synthetic_generation.grammar import BasicTransactionGrammar
from synthetic_generation.sampling import (
    apply_adaptive_plausibility,
    build_contrastive_scores,
    mask_to_allowed,
    top_p_sample,
)


logger = logging.getLogger(__name__)


@dataclass
class GeneratedUser:
    synthetic_user_id: str
    input_ids: list[int]
    attention_mask: list[int]
    num_tokens: int
    num_transactions: int
    target_transactions: int
    stopped_reason: str


def _forward_next_log_probs(model: torch.nn.Module, input_ids: list[int], *, device: torch.device, dtype: str) -> torch.Tensor:
    if not input_ids:
        raise ValueError("Cannot run autoregressive generation with an empty prefix. Provide a BOS token id.")

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    use_cuda_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_cuda_autocast):
        output = model(input_ids=x)

    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, tuple):
        logits = output[0]
    if logits is None:
        raise RuntimeError(f"Model output has no logits; got type={type(output).__name__}")
    return torch.log_softmax(logits[0, -1, :].float(), dim=-1)


def generate_one_user(
    *,
    config: GenerationConfig,
    teacher: torch.nn.Module,
    target: torch.nn.Module | None,
    grammar: BasicTransactionGrammar,
    target_transactions: int,
    sample_idx: int,
    torch_generator: torch.Generator,
    initial_sequence: list[int] | None = None,
) -> GeneratedUser:
    device = next(teacher.parameters()).device
    # Prefer an explicit real-prefix seed (this data has no BOS); fall back to the grammar's
    # BOS-based start when no seed is supplied.
    sequence = list(initial_sequence) if initial_sequence else grammar.initial_sequence()
    if not sequence:
        raise ValueError(
            "Generation from zero requires a real-prefix seed or a BOS/control token for this "
            "model. Pass --seed-source (recommended) or --bos-token-id."
        )

    max_total_tokens = max(
        config.max_context_tokens,
        target_transactions * config.max_tokens_per_transaction + len(sequence) + 8,
    )
    stopped_reason = "target_transactions"

    while not grammar.should_stop(sequence, target_transactions):
        if len(sequence) >= max_total_tokens:
            stopped_reason = "max_total_tokens"
            break

        context = sequence[-config.max_context_tokens :]
        allowed = grammar.allowed_tokens(sequence, device=device)
        teacher_log_probs = _forward_next_log_probs(teacher, context, device=device, dtype=config.dtype)

        if config.mode == "vanilla":
            scores = teacher_log_probs
        else:
            if target is None:
                raise RuntimeError("contrastive mode requires target model")
            target_device = next(target.parameters()).device
            if target_device != device:
                raise RuntimeError(f"Teacher and target must be on same device; got {device} and {target_device}")
            target_log_probs = _forward_next_log_probs(target, context, device=device, dtype=config.dtype)
            if target_log_probs.shape != teacher_log_probs.shape:
                raise RuntimeError(
                    "Teacher and target vocabularies differ: "
                    f"{teacher_log_probs.shape} vs {target_log_probs.shape}"
                )
            scores = build_contrastive_scores(
                teacher_log_probs=teacher_log_probs,
                target_log_probs=target_log_probs,
                gamma=config.gamma,
            )
            scores = apply_adaptive_plausibility(
                scores=scores,
                teacher_log_probs=teacher_log_probs,
                allowed_token_ids=allowed,
                alpha=config.alpha,
            )

        if config.mode == "vanilla":
            scores = mask_to_allowed(scores, allowed)
        token_id = top_p_sample(scores, temperature=config.temperature, top_p=config.top_p, generator=torch_generator)
        sequence.append(token_id)

    if grammar.eos_token_id is not None and sequence and sequence[-1] == int(grammar.eos_token_id):
        stopped_reason = "eos"

    # Training rows should contain only the synthetic history. If generation
    # started from a BOS control token, keep it because the model conditioned on
    # that grammar; future strict FSMs can decide whether to strip it.
    n_txn = grammar.count_transactions(sequence)
    return GeneratedUser(
        synthetic_user_id=f"synth_{sample_idx:012d}",
        input_ids=[int(x) for x in sequence],
        attention_mask=[1] * len(sequence),
        num_tokens=len(sequence),
        num_transactions=n_txn,
        target_transactions=target_transactions,
        stopped_reason=stopped_reason,
    )

