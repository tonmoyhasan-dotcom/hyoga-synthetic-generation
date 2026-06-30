from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def setup_core_imports() -> None:
    experiment_dir = Path(__file__).resolve().parents[1]
    src_dir = experiment_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from core.utils import setup_python_path_for_monorepo_imports

    setup_python_path_for_monorepo_imports(build_schemas=False)


@dataclass
class LoadedGeneratorModel:
    model: torch.nn.Module
    tokenizer: object
    metadata: dict
    ckpt_path: str


def load_lightning_generator_model(
    *,
    ckpt_path: str,
    context_length: int | None,
    device: str,
) -> LoadedGeneratorModel:
    """Load a Hyoga Lightning checkpoint for token-level generation."""
    setup_core_imports()
    from core.evaluation.loader import load_lightning_core_model

    wrapper, _preprocessor, metadata = load_lightning_core_model(
        lightning_ckpt_path=ckpt_path,
        preprocessing_component_paths=None,
        context_length=context_length,
    )
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    if model is None:
        raise RuntimeError(f"Loaded wrapper from {ckpt_path} has no model")
    if tokenizer is None:
        raise RuntimeError(f"Loaded wrapper from {ckpt_path} has no tokenizer")

    torch_device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    if str(torch_device) != device:
        logger.warning("Requested device %s unavailable; using %s", device, torch_device)
    model.to(torch_device)
    model.eval()

    logger.info(
        "Loaded generator model from %s: class=%s context=%s vocab=%s",
        ckpt_path,
        metadata.get("model_class"),
        metadata.get("max_sequence_length"),
        metadata.get("vocab_size"),
    )
    return LoadedGeneratorModel(model=model, tokenizer=tokenizer, metadata=metadata, ckpt_path=ckpt_path)

