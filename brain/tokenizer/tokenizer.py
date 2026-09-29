"""UTF-8 byte tokenizer for Raksam.

A byte vocabulary avoids the character-level tokenizer's unknown-character
problem and naturally supports English, Bangla, emojis, device identifiers,
JSON action schemas, and future languages without external tokenizers.
"""
from __future__ import annotations
import json

class CharTokenizer:
    # Kept under the old class name so existing imports remain compatible.
    BYTE_COUNT = 256
    SPECIAL = {"<pad>": 256, "<bos>": 257, "<eos>": 258}

    def __init__(self, tokenizer_path: str = "brain/tokenizer/tokenizer.json"):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.pad_id = int(self.stoi["<pad>"])
        self.bos_id = int(self.stoi["<bos>"])
        self.eos_id = int(self.stoi["<eos>"])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="replace")
