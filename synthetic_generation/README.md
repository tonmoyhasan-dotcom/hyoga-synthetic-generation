# Synthetic Hyoga User Generation

This package is the first generator-side implementation for synthetic transaction
histories. It is intentionally isolated so it can later move into a dedicated
synthetic-data-generation + verifier repo.

## Scope

Implemented:

- Vanilla autoregressive generation from a stronger Hyoga teacher `G`.
- Contrastive generation against a frozen reference `M0`.
- Generation from a zero/user-start sequence.
- Target transaction-count schedules.
- Sliding context for 4k models while storing longer generated histories.
- Parquet output with synthetic metadata.
- V0 grammar that tracks transaction separators and blocks obvious label/control tokens.

Not implemented yet:

- Strict field-level transaction FSM.
- Verifier gates.
- Synthetic classification supervision.

Synthetic rows currently include a dummy `defaults_6m = 0` only for schema compatibility.
They should be used as NTP-only rows unless downstream training honors
`classification_loss_mask = 0`.

## Recommended First Experiment

Use:

- `G`: 4k pretrained Hyoga trained on 40M real data.
- `M0`: 4k pretrained Hyoga trained on 2M real data.

Compare:

- `2M real only`
- `2M real + vanilla G synthetic`
- `2M real + contrastive G-vs-M0 synthetic`
- `40M real` reference

The known 40M 4k pretrained candidate is:

```text
run_id: c9de3c3bfe604873adb7551adf72d04a
ckpts:  gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/checkpoints/
config: gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/config/resolved_config.yaml
```

Its resolved config has:

```yaml
collate_fn.init_args.max_sequence_length: 4096
model.init_args.sequence_length: 4096
```

## Smoke Test

From `research/experiments/core_v1`:

```bash
uv run --inexact --no-default-groups --group unsloth_pt29_cudnn919 \
  python -m synthetic_generation.smoke_test \
  --output-path /tmp/hyoga_synth_smoke \
  --mode contrastive \
  --num-samples 4 \
  --target-transactions 3
```

## Real Checkpoint Generation

From `research/experiments/core_v1`:

```bash
uv run --inexact --no-default-groups --group unsloth_pt29_cudnn919 \
  python -m synthetic_generation.cli \
  --mode contrastive \
  --teacher-run-id c9de3c3bfe604873adb7551adf72d04a \
  --teacher-ckpt gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/checkpoints/step=078113.ckpt \
  --target-run-id <2m_pretrained_run_id> \
  --target-ckpt <2m_pretrained_ckpt> \
  --output-path /tmp/hyoga_synth_contrastive_small \
  --lengths 100,250,500 \
  --samples-per-length 10 \
  --max-transactions 2250 \
  --max-context-tokens 4096 \
  --max-tokens-per-transaction 12 \
  --temperature 0.9 \
  --top-p 0.95 \
  --alpha 0.05 \
  --gamma 0.5 \
  --seed 42 \
  --shard-size 100
```

For raw teacher sampling, use:

```bash
--mode vanilla --gamma 0
```

and omit `--target-run-id` / `--target-ckpt`.

## Output Columns

Each row is one synthetic user snapshot:

- `synthetic_user_id`
- `input_ids`
- `attention_mask`
- `num_tokens`
- `num_transactions`
- `target_transactions`
- `stopped_reason`
- `source`
- `generation_mode`
- `teacher_run_id`
- `target_run_id`
- `teacher_ckpt`
- `target_ckpt`
- `temperature`
- `top_p`
- `alpha`
- `gamma`
- `seed`
- `classification_loss_mask`
- `ntp_loss_mask`
- `defaults_6m`

## Notes

The V0 generator counts transactions via the tokenizer separator token. If a model
does not expose a valid BOS token, pass `--bos-token-id`. If it does not expose a
valid separator token, pass `--sep-token-id`.

The current grammar is a safe bootstrap, not the final grammar-constrained decoding
described in the research plan. The next generator milestone should replace
`BasicTransactionGrammar` with a field-aware FSM over the actual Hyoga transaction
token template.

