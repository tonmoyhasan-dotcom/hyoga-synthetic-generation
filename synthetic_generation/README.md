# Synthetic transaction-sequence generation (HFv3 / Hyoga) — first implementation

Generate synthetic tokenized user histories with a frozen **teacher G** (40M-pretrained 4k
Qwen3-GDN), optionally guided contrastively by a frozen **target M0** (same architecture,
2M-pretrained 4k). Synthetic rows are **NTP-only** (`ntp_loss_mask=1`,
`classification_loss_mask=0`, dummy `defaults_6m`); never train classification on them.

## Models / run IDs

| role | run | checkpoint | notes |
|------|-----|-----------|-------|
| **G** (teacher, 40M) | `c9de3c3bfe604873adb7551adf72d04a` (`pretrain-...-40M-ra-nca-pop-scr-spc-...`) | `.../checkpoints/step=078113.ckpt` (last) | seq_len 4096, vocab 16386, label tokens 16384/16385, SEP=2 |
| **M0** (target, 2M) | `7d68a2cb57464f47b5055dd67db98c98` (`hfv3_pretrain_liccdctrplbcit_dsw1`) | `.../checkpoints/step=007809.ckpt` (last) or `.../checkpoints/best-val-auc-step=007805.ckpt` | seq_len 4096, vocab 16386, train path contains `train_2M` |

Config confirmed from G's `resolved_config.yaml`: model `Qwen3GDNModel` (28 layers, d_model 1024),
collate `CollateWithPromptAndBinaryLabels` (`max_sequence_length=4096`, `label_column=defaults_6m`),
data `…/LiCcDcTrPlBcItScrSpc/tokenized/v0/{train_40m,validation_100k}`.

## Data format notes
- A row's `input_ids` is a concatenation of transactions, each ending in **SEP=2**. There is
  **no BOS** and **no EOS distinct from SEP**, so:
  - generation is **seeded from a real prefix** (`--seed-source`, first N real transactions),
  - stopping is purely by **transaction count** (SEP is *not* treated as EOS).
- Label tokens 16384/16385 are appended by the collate at train time, **not** stored in
  `input_ids`; they are banned during generation.

## Run — vanilla teacher (works today)
```bash
cd research/experiments/core_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" .venv/bin/python -m synthetic_generation.cli \
  --mode vanilla \
  --teacher-ckpt gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/checkpoints/step=078113.ckpt \
  --teacher-run-id c9de3c3bfe604873adb7551adf72d04a \
  --output-path /tmp/syn_smoke2 \
  --num-samples 4 --lengths 20 \
  --seed-source "aic-br-us-central1-prod-workstream-credit-adhoc/hyoga_foundation_v3/ra_nca/v1/main/pre_processed/LiCcDcTrPlBcItScrSpc/tokenized/v0/validation_100k/*.parquet" \
  --seed-prefix-transactions 1 --max-context-tokens 4096 --grammar-mode basic --device cuda --seed 0
```

## Run — contrastive teacher-student
```bash
  ... --mode contrastive \
  --target-ckpt gs://aic-prod-mlflow/1536/7d68a2cb57464f47b5055dd67db98c98/artifacts/checkpoints/step=007809.ckpt \
  --target-run-id 7d68a2cb57464f47b5055dd67db98c98 --gamma 0.5   # sweep gamma in {0.2,0.5,1.0}
```
`guided = logits_G + gamma*(logits_G - logits_M0)`, with adaptive-plausibility (`--alpha`) keeping
samples within an alpha-factor of the teacher's best allowed token.

## Grammar constraints
`--grammar-mode basic` is the default. It applies the current lightweight token mask: block pad/BOS
and label/prediction tokens, treat SEP as the transaction boundary, and disable EOS when EOS collides
with SEP. This is the recommended safe mode.

`--grammar-mode none` disables token masking and lets the model sample from the full vocabulary. The
loop still counts SEP tokens to decide when the requested transaction length has been reached, but no
tokens are forbidden. Use this for ablations only; it can emit label/control/pad tokens if the model
assigns them probability.

## Smoke test
```bash
cd research/experiments/core_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" .venv/bin/python -m synthetic_generation.smoke_test \
  --parquet /tmp/syn_smoke2 \
  --teacher-ckpt gs://aic-prod-mlflow/1536/c9de3c3bfe604873adb7551adf72d04a/artifacts/checkpoints/step=078113.ckpt
```
Checks: token ids in vocab, snapshots end in SEP, NTP-only flags, non-empty/distinct, max len
<= context, and that the training `collate_fn` batches the rows and the model forward runs.

## Output schema (per row)
`synthetic_user_id, input_ids, attention_mask, num_tokens, num_transactions,
target_transactions, stopped_reason, source, generation_mode, teacher_run_id, target_run_id,
teacher_ckpt, target_ckpt, temperature, top_p, alpha, gamma, grammar_mode, seed, classification_loss_mask=0,
ntp_loss_mask=1, defaults_6m=0 (DUMMY)`. A `generation_config.json` is written alongside.

## Experiment framing
- A. M trained on 2M real only.
- B. M trained on 2M real + synthetic from G_40M.
- C. M trained on 40M real — **oracle/reference baseline** (not a theoretical upper bound).

## Known gaps / TODOs
- **A1 precondition** (`NTP_loss(G_40M) < NTP_loss(M0_2M)` on real validation): needs a
  validation-NTP eval for both before trusting G as generator.
- **Scale/perf**: generation re-forwards the full context each step (O(L^2)); add KV-cache/state
  reuse before large-scale runs.
- **Grammar**: V0 `basic` only bans label/pad/control tokens and uses SEP as the boundary; `none`
  disables masking for ablations. A field-level FSM can replace `BasicTransactionGrammar` without
  touching the loop.
- **Data-module path**: rows are HFv3-shaped and collate/forward-validated; use
  `--index-litdata` to produce a LitData index if feeding `LitDataStreamingDataModule` directly.
- **Vanilla repetition**: gamma=0 samples are quite repetitive (the cross-snapshot redundancy the
  project targets); contrastive decoding is the intended novelty lever.
