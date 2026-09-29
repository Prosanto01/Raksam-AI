# Raksam Brain V3

Created by: **Prosanto Raksam**

## Fixed from V2
- Safe UTF-8 output validation and clean fallback instead of garbled/undefined output.
- Model/tokenizer vocabulary compatibility checks.
- Context-window protection.
- Repetition penalty and safer sampling.
- Weighted training examples are now actually used.
- Candidate training remains separate from the active brain.
- Evaluation entry point added.
- Model metadata is stored in checkpoints.

## Self-learning foundation
Conversation feedback, successful actions, approved experiences, replay data, and approved web knowledge can feed future candidate training. Candidates should be evaluated before promotion.

## Important
A model trained from scratch with only a few hundred examples cannot be expected to have broad knowledge. V3 therefore focuses on a correct learning/evaluation pipeline first; capability grows with the quality and amount of training data.

## Commands
`py -3.13 chat.py`
`py -3.13 -m brain.train --epochs 1 --batch-size 2`
`py -3.13 evaluate.py`
