"""Synthetic transaction-sequence generation for HFv3 / NuFormer (first implementation).

Two frozen pretrained 4k Hyoga (Qwen3-GDN) models drive generation:
  * G  = stronger teacher pretrained on 40M real rows (run c9de3c3b...).
  * M0 = weaker target/student pretrained on 2M real rows (same architecture; run id TBD).

Decoding modes:
  * raw teacher          : sample autoregressively from G (gamma == 0).
  * contrastive teacher-student :
        guided_logits = logits_G + gamma * (logits_G - logits_M0)
    favouring continuations the 40M teacher models better than the 2M target.

Synthetic rows are NTP-only: `defaults_6m` is a dummy and classification loss must be
masked (ntp_loss_mask=1, classification_loss_mask=0).
"""
