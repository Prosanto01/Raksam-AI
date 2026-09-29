"""Fast, offline structural verification for the configured 52M RB3 architecture."""
from __future__ import annotations

import yaml
import torch

from brain.dataset import UnifiedDataset, action_prompt, chat_prompt, knowledge_prompt, reflection_prompt
from brain.model import BrainConfig, BrainModel
from brain.tokenizer.tokenizer import CharTokenizer

EXPECTED_PARAMETERS = 52_460_544

with open("config/settings.yaml", encoding="utf8") as file:
    architecture = yaml.safe_load(file)["brain"]["architecture"]
tokenizer = CharTokenizer()
model = BrainModel(BrainConfig(vocab_size=tokenizer.vocab_size, **architecture))
assert model.num_parameters() == EXPECTED_PARAMETERS, model.num_parameters()
assert len(UnifiedDataset(tokenizer, block_size=architecture["block_size"])) == 4608
for prompt in (chat_prompt("Hello"), action_prompt("GOAL: open Notepad\nDEVICE: this_pc"), knowledge_prompt("Example", "https://example.invalid"), reflection_prompt("open Notepad", "success")):
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)[-32:]], dtype=torch.long)
    logits, loss = model(ids)
    assert logits.shape == (1, ids.shape[1], tokenizer.vocab_size) and loss is None
print(f"PASS: configured RB3 has {model.num_parameters():,} parameters and accepts all four unified modalities.")
