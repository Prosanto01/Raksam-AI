# Raksam AI V3 — Raksam Brain V3 (RB3) training status

Created by: Prosanto Raksam

## Unified architecture

RB3 is one causal Transformer backbone for four explicitly marked input types:

- `<CHAT>` for conversation and memory-conditioned replies.
- `<TASK>` for device plans and structured actions.
- `<KNOWLEDGE>` for approved source material.
- `<REFLECTION>` for reviewed action/conversation outcomes.

Screen, OCR, UI-automation and device observations remain safely serialized by the perception layer before reaching the backbone. This is an interface-level multimodal design, not a pixel-trained vision model; no image training corpus is included in this archive.

The configured RB3 architecture has 8 layers, 12 query heads, 4 KV heads (GQA), 768 hidden dimensions, a 2.8x SwiGLU MLP, 128-token context, RoPE, RMSNorm, residual scaling, causal attention and tied embeddings. With the included 259-symbol tokenizer its exact trainable parameter count is **52,460,544**.

## Data inventory at verification

The unified dataset loader accepts **4,608** usable examples: 4,466 conversational records, 3 device conversation records, 109 action demonstrations and 30 approved internet-learning records. Evaluation, raw event and episodic-memory JSONL files are intentionally not used for direct supervised training unless their records are explicitly approved.

## Checkpoint policy

`brain/checkpoints/latest.pt` remains the tested 0.40M compatibility checkpoint. It is not an RB3-52M checkpoint and is retained so existing chat/action behaviour stays available.

Any 52M training attempt is written only to `brain/checkpoints/candidate.pt`. Candidate metadata records exact steps, target-token count, train/validation loss, whether validation was sampled, and whether the epoch was partial. Do not promote it as a completed model without a full run and evaluation.

## Compute limitation

A 52M model trained from scratch needs substantially more data and compute than this 4-core CPU environment provides for a complete useful run. The included candidate is a short, explicitly marked smoke-training run only if `candidate.pt` is present. It is **not fully trained** and must not replace the active checkpoint.

For a real run, use a GPU and a much larger licensed/curated dataset, then evaluate held-out conversation and action tasks before promotion.
