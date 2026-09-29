# Raksam Autonomous Self-Learning

Raksam can now operate as a continual-learning system without Ollama, LM Studio,
OpenAI, or another AI model at runtime.

## What happens automatically

1. Raksam visits only the public URLs/domains in `config/web_learning.yaml`.
2. It respects `robots.txt`, rate limits, size limits and content-type checks.
3. New material is deduplicated and stored with provenance.
4. Once `min_new_documents` is reached, Raksam trains its own transformer.
5. Training creates a candidate checkpoint rather than corrupting the active brain.
6. If training succeeds and `auto_promote: true`, the candidate becomes active and the old checkpoint is kept as `brain/checkpoints/previous.pt`.
7. The next cycle learns from additional data and repeats the process.

## Start autonomous learning

Edit the allowlist first, then:

```bash
python self_learn.py --loop 3600
```

Or use the configured interval:

```bash
python self_learn.py --loop 3600
```

## Important distinction

This is **continual self-training**, not magical self-improvement. The model can
only learn patterns present in its data and training objective. Internet data can
contain errors, bias, spam and adversarial text, so the allowlist and provenance
system are deliberately retained.

Raksam can also learn from its own interaction experience in a future extension:
store successful/failed tasks, user corrections and device outcomes as replay data,
then periodically train on those experiences.
