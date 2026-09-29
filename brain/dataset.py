"""Unified causal-LM dataset for Raksam's own brain.

The dataset intentionally uses explicit task markers. This teaches one model
to distinguish conversation, planning, action/tool calls, knowledge and
reflection instead of creating disconnected models.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from brain.tokenizer.tokenizer import CharTokenizer


DEMO_DIR = Path("data/demonstrations")
CHAT_DIR = Path("data/conversations")
WEB_DIR = Path("data/web_knowledge")
EXP_DIR = Path("data/experiences")


def action_prompt(prompt: str) -> str:
    return "<TASK>\n" + prompt.rstrip() + "\n<ASSISTANT_ACTION>\n"


def chat_prompt(user: str, history=None) -> str:
    lines = ["<CHAT>"]

    for item in (history or [])[-12:]:
        # Accept both training dictionaries and runtime Turn objects.
        if isinstance(item, dict):
            role = str(item.get("role", "user")).upper()
            content = str(item.get("content", ""))
        else:
            role = str(getattr(item, "role", "user")).upper()
            content = str(getattr(item, "content", ""))
        if content.strip():
            lines.append(f"{role}: {content}")

    lines += [
        f"USER: {user}",
        "ASSISTANT:",
    ]

    return "\n".join(lines) + "\n"


def knowledge_prompt(title: str, source: str) -> str:
    result = "<KNOWLEDGE>\n"

    if title:
        result += f"TITLE: {title}\n"

    if source:
        result += f"SOURCE: {source}\n"

    result += "TEXT:\n"
    return result


def reflection_prompt(goal: str, result: str) -> str:
    return (
        "<REFLECTION>\n"
        f"GOAL: {goal}\n"
        f"RESULT: {result}\n"
        "LESSON:\n"
    )


class UnifiedDataset(Dataset):
    """Single dataset for conversation, actions, knowledge and experience."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        block_size: int = 2048,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.examples = []
        self.rng = random.Random(seed)

        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")

        self._load_chats()
        self._load_actions()
        self._load_web()
        self._load_reflections()

        self.rng.shuffle(self.examples)

    def _add(self, prompt, target, weight=1.0):
        """Add a prompt/target pair while guaranteeing learnable labels.

        The old implementation could truncate the entire target when the
        prompt was longer than block_size. That produced labels containing
        only -1, causing cross_entropy() to return NaN.

        This implementation always reserves at least one token for the target.
        """

        prompt = str(prompt or "")
        target = str(target or "")

        if not prompt or not target:
            return

        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_bos=True,
        )

        target_tokens = self.tokenizer.encode(
            target,
            add_bos=False,
            add_eos=True,
        )

        if not target_tokens:
            return

        # Reserve at least one position for the target.
        max_prompt_tokens = self.block_size - 1

        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[:max_prompt_tokens]

        available_target_tokens = self.block_size - len(prompt_tokens)

        if available_target_tokens <= 0:
            return

        target_tokens = target_tokens[:available_target_tokens]

        ids = prompt_tokens + target_tokens
        labels = [-1] * len(prompt_tokens) + target_tokens

        # Safety check: never add an example without a learnable target.
        valid_targets = sum(label >= 0 for label in labels)

        if len(ids) >= 2 and valid_targets > 0:
            self.examples.append(
                (
                    ids,
                    labels,
                    max(0.1, float(weight)),
                )
            )

    def _jsonl(self, path: Path):
        """Safely read JSONL records."""

        if not path.exists():
            return

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                    if isinstance(record, dict):
                        yield record

                except json.JSONDecodeError:
                    # Ignore malformed individual records rather than
                    # destroying the complete training run.
                    continue

    def _load_chats(self):
        """Load conversation examples."""

        CHAT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in CHAT_DIR.glob("*.jsonl"):
            for record in self._jsonl(path):

                user = str(
                    record.get("user", "")
                ).strip()

                assistant = str(
                    record.get("assistant", "")
                ).strip()

                if not user or not assistant:
                    continue

                quality = record.get("quality")

                weight = (
                    1.5
                    if quality in ("gold", "approved")
                    else 1.0
                )

                self._add(
                    chat_prompt(
                        user,
                        record.get("history", []),
                    ),
                    assistant,
                    weight,
                )

    def _load_actions(self):
        """Load computer/device action demonstrations."""

        DEMO_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in DEMO_DIR.glob("*.jsonl"):
            for record in self._jsonl(path):

                prompt = str(
                    record.get("prompt_text", "")
                ).strip()

                action = dict(
                    record.get("action", {})
                )

                if record.get("device"):
                    action.setdefault(
                        "device",
                        record["device"],
                    )

                if not prompt or not action:
                    continue

                target = json.dumps(
                    action,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                self._add(
                    action_prompt(prompt),
                    target,
                    1.4,
                )

    def _load_web(self):
        """Load approved web knowledge."""

        WEB_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in WEB_DIR.glob("*.jsonl"):
            for record in self._jsonl(path):

                text = str(
                    record.get("text", "")
                ).strip()

                title = str(
                    record.get("title", "")
                ).strip()

                source = str(
                    record.get("source_url", "")
                ).strip()

                if len(text) < 300:
                    continue

                # Keep chunks reasonably sized for the selected context.
                chunk_chars = max(
                    1500,
                    self.block_size * 3,
                )

                for start in range(
                    0,
                    len(text),
                    chunk_chars,
                ):
                    chunk = text[
                        start:start + chunk_chars
                    ].strip()

                    if len(chunk) < 300:
                        continue

                    self._add(
                        knowledge_prompt(
                            title,
                            source,
                        ),
                        chunk,
                        0.45,
                    )

    def _load_reflections(self):
        """Load approved experiences and lessons."""

        approved_path = EXP_DIR / "approved.jsonl"

        for record in self._jsonl(approved_path):

            payload = record.get(
                "payload",
                {},
            )

            if not isinstance(payload, dict):
                continue

            kind = record.get("kind")

            # Successful action experience.
            if (
                kind == "action"
                and payload.get("goal")
                and payload.get("result")
            ):
                action = payload.get(
                    "action",
                    {},
                )

            # User-approved live agent steps. These preserve the observation
            # that existed before the action, so they teach the brain a real
            # screen-state-to-action mapping rather than only a final result.
            elif (
                kind == "agent_step"
                and payload.get("goal")
                and payload.get("action")
                and payload.get("before_observation")
            ):
                prompt = (
                    f"GOAL: {payload['goal']}\n"
                    f"DEVICE: {payload.get('device', 'this_pc')}\n"
                    f"CURRENT_OBSERVATION:\n{payload['before_observation']}\n"
                    "Choose ONE next action."
                )
                self._add(
                    action_prompt(prompt),
                    json.dumps(payload["action"], ensure_ascii=False, separators=(",", ":")),
                    1.15,
                )

                lesson = (
                    "Successful execution lesson: "
                    + json.dumps(
                        action,
                        ensure_ascii=False,
                    )
                )

                self._add(
                    reflection_prompt(
                        payload["goal"],
                        payload["result"],
                    ),
                    lesson,
                    0.9,
                )

            # Approved conversation experience.
            elif (
                kind == "conversation"
                and payload.get("user")
                and payload.get("assistant")
            ):
                self._add(
                    chat_prompt(
                        payload["user"],
                        payload.get(
                            "history",
                            [],
                        ),
                    ),
                    payload["assistant"],
                    1.3,
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids, labels, _weight = self.examples[index]

        # Input predicts the next token.
        x = ids[:-1]

        # Shifted target.
        y = labels[1:]

        # Pad to block size.
        padding = self.block_size - len(x)

        if padding > 0:
            x = x + [
                self.tokenizer.pad_id
            ] * padding

            y = y + [
                -1
            ] * padding

        else:
            x = x[:self.block_size]
            y = y[:self.block_size]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(float(_weight), dtype=torch.float32),
        )


# Backwards-compatible alias.
DemoDataset = UnifiedDataset
