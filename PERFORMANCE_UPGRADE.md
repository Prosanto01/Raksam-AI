# RB3-52M context and output upgrade

The configured Raksam Brain V3 (RB3) architecture remains a 52.46M-parameter
model. Its next training run is now configured for a 512-token context window,
up from 128 tokens, with up to 384 generated tokens for conversational replies.

The context setting does not add trainable parameters because RB3 uses rotary
position embeddings. It does increase attention memory and CPU/GPU time,
roughly with the square of sequence length. A GPU is strongly recommended for
a complete 52M training run at this context length.

## Compatibility

No checkpoint was removed or overwritten. Existing checkpoints keep the
configuration saved inside them: the included 52M candidate has a 128-token
context and the compatibility checkpoint remains unchanged. Train a new
candidate to use the 512-token setting:

```powershell
py -3.13 brain/train.py --candidate --epochs 1
```

The normal training command first performs the configured bounded public-web
collection pass. Add `--skip-web-collection` when working offline.

Evaluate the new candidate before promotion; longer context does not itself
make an under-trained model more knowledgeable.
