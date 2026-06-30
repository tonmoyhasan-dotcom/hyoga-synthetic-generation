from __future__ import annotations

import torch


NEG_INF = -torch.inf


def mask_to_allowed(scores: torch.Tensor, allowed_token_ids: torch.Tensor) -> torch.Tensor:
    masked = torch.full_like(scores, NEG_INF)
    masked[..., allowed_token_ids] = scores[..., allowed_token_ids]
    return masked


def apply_adaptive_plausibility(
    *,
    scores: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Keep tokens within an ``alpha`` probability factor of the teacher's best allowed token."""
    allowed_teacher = teacher_log_probs[..., allowed_token_ids]
    threshold = allowed_teacher.max(dim=-1, keepdim=True).values + torch.log(
        torch.tensor(alpha, device=teacher_log_probs.device, dtype=teacher_log_probs.dtype)
    )
    keep_allowed = allowed_teacher >= threshold

    masked = torch.full_like(scores, NEG_INF)
    allowed_scores = scores[..., allowed_token_ids]
    masked[..., allowed_token_ids] = torch.where(keep_allowed, allowed_scores, torch.full_like(allowed_scores, NEG_INF))
    return masked


def top_p_sample(scores: torch.Tensor, *, temperature: float, top_p: float, generator: torch.Generator) -> int:
    """Sample one token from a 1D score vector using temperature and nucleus filtering."""
    if scores.dim() != 1:
        raise ValueError(f"top_p_sample expects 1D scores, got shape={tuple(scores.shape)}")

    scaled = scores / temperature
    finite = torch.isfinite(scaled)
    if not finite.any():
        raise RuntimeError("No finite token scores remain after masking")

    sorted_scores, sorted_indices = torch.sort(scaled, descending=True)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    keep = cumulative <= top_p
    keep[0] = True
    filtered_scores = torch.full_like(sorted_scores, NEG_INF)
    filtered_scores[keep] = sorted_scores[keep]
    filtered_probs = torch.softmax(filtered_scores, dim=-1)
    sampled_pos = torch.multinomial(filtered_probs, num_samples=1, generator=generator)
    return int(sorted_indices[sampled_pos.item()].item())


def build_contrastive_scores(
    *,
    teacher_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    return teacher_log_probs + gamma * (teacher_log_probs - target_log_probs)

