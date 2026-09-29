# Raksam Brain v2

This release changes the model itself rather than only adding wrappers around the old transformer.

## Core design
- One causal transformer for conversation, knowledge, planning traces and device/tool actions.
- GQA reduces KV-cache/memory cost while retaining multiple query heads.
- RoPE handles position without a learned absolute position table.
- RMSNorm + SwiGLU + residual scaling improve optimization stability.
- 2048-token context is the new default and is configurable.
- Tied embeddings reduce parameter count.
- A lightweight episodic memory store retrieves relevant past experiences without another AI model.

## Scaling
The default 16-layer/640-dimension configuration is a practical research starting point. For serious pretraining, scale layers, width, context, data and compute together. A larger architecture without proportionally larger data/compute will not automatically become smarter.

## Continual learning
Raw experience is quarantined. Approved/high-trust examples enter replay. Training writes a candidate checkpoint and validation loss is recorded before promotion. Keep the previous checkpoint for rollback.

## Important limitation
Self-training cannot magically create frontier intelligence. Raksam needs large, diverse, high-quality training corpora, tokenizer quality, substantial GPU compute, careful evaluation, and a curriculum covering language, reasoning, tool use, multimodal/device tasks and corrections.
