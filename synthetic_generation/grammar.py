from __future__ import annotations

from dataclasses import dataclass

import torch


_LABEL_TOKEN_STRINGS = (
    "<DEFAULT_SCORE:DEFAULT>",
    "<DEFAULT_SCORE:NOT_DEFAULT>",
    "<|default|>",
    "<|not_default|>",
    "<|predict_default|>",
)


@dataclass
class BasicTransactionGrammar:
    """Minimal token grammar for tokenized transaction histories.

    This V0 grammar is intentionally conservative: it lets the trained Hyoga model
    provide the transaction-token distribution, while disallowing obvious control
    and label tokens and using the separator token as the transaction boundary.
    A stricter field-level FSM can replace this class without changing the
    generation loop.
    """

    vocab_size: int
    sep_token_id: int
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    disallowed_token_ids: frozenset[int] = frozenset()
    constrain_tokens: bool = True

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        vocab_size: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        sep_token_id: int | None = None,
        disallow_label_tokens: bool = True,
        constrain_tokens: bool = True,
    ) -> "BasicTransactionGrammar":
        resolved_vocab_size = int(vocab_size or len(tokenizer))
        resolved_sep = sep_token_id
        if resolved_sep is None:
            resolved_sep = tokenizer.sep_token_id
        if resolved_sep is None:
            resolved_sep = tokenizer.convert_tokens_to_ids("[SEP]")
        if resolved_sep is None or resolved_sep < 0 or resolved_sep == tokenizer.unk_token_id:
            # The RA_NCA tokenizer used by the dedup preprocessor ends each
            # transaction span with id 2. Keep this as the final fallback.
            resolved_sep = 2

        resolved_bos = tokenizer.bos_token_id if bos_token_id is None else bos_token_id
        resolved_eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        # This tokenizer reuses the SEP id (2) as eos. SEP is a transaction boundary, not an
        # end-of-history marker, so treating it as EOS would stop generation after the first
        # transaction. Drop EOS when it collides with SEP; stopping is by transaction count.
        if resolved_eos is not None and int(resolved_eos) == int(resolved_sep):
            resolved_eos = None

        disallowed: set[int] = set()
        if constrain_tokens and tokenizer.pad_token_id is not None:
            disallowed.add(int(tokenizer.pad_token_id))
        if constrain_tokens and disallow_label_tokens:
            for token_str in _LABEL_TOKEN_STRINGS:
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
                    disallowed.add(int(token_id))

        if constrain_tokens and resolved_bos is not None:
            disallowed.add(int(resolved_bos))

        return cls(
            vocab_size=resolved_vocab_size,
            sep_token_id=int(resolved_sep),
            bos_token_id=resolved_bos,
            eos_token_id=resolved_eos,
            pad_token_id=tokenizer.pad_token_id,
            disallowed_token_ids=frozenset(disallowed),
            constrain_tokens=constrain_tokens,
        )

    def initial_sequence(self) -> list[int]:
        return [] if self.bos_token_id is None else [int(self.bos_token_id)]

    def count_transactions(self, token_ids: list[int]) -> int:
        return sum(1 for token_id in token_ids if int(token_id) == self.sep_token_id)

    def just_finished_transaction(self, token_id: int) -> bool:
        return int(token_id) == self.sep_token_id

    def should_stop(self, token_ids: list[int], target_transactions: int) -> bool:
        if self.count_transactions(token_ids) >= target_transactions:
            return True
        return bool(token_ids and self.eos_token_id is not None and int(token_ids[-1]) == int(self.eos_token_id))

    def allowed_tokens(self, token_ids: list[int], *, device: torch.device) -> torch.Tensor:
        if not self.constrain_tokens:
            return torch.arange(self.vocab_size, dtype=torch.long, device=device)

        allowed = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        if self.disallowed_token_ids:
            ids = [token_id for token_id in self.disallowed_token_ids if 0 <= token_id < self.vocab_size]
            if ids:
                allowed[torch.tensor(ids, dtype=torch.long, device=device)] = False
        # EOS is not allowed before the requested number of transactions.
        if (
            self.eos_token_id is not None
            and 0 <= int(self.eos_token_id) < self.vocab_size
            and self.count_transactions(token_ids) == 0
        ):
            allowed[int(self.eos_token_id)] = False
        return allowed.nonzero(as_tuple=False).flatten()

