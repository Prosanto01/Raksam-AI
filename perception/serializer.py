"""
perception/serializer.py

Builds the exact text block your BRAIN model reads as input.
Keeping this in one place means recorder/, brain/train.py, and main.py
all format state the same way - critical, since the brain is trained
on this exact text shape.

Format:
    GOAL: <task goal>
    STATE:
    WINDOW: <active window title>
    ELEMENTS:
      [Button "OK" @ (120,300)]
      [Edit "Filename" @ (400,150)]
    OCR_TEXT: <fallback text, if any>
    HISTORY:
      1. click("OK")
      2. type("hello.txt")
    ACTION:
"""

from __future__ import annotations

from perception.screen_state import ScreenState


def build_prompt(goal: str, state: ScreenState, history: list[str], max_history: int = 5) -> str:
    parts = [f"GOAL: {goal}", "STATE:", state.to_prompt_text(), "HISTORY:"]

    recent = history[-max_history:] if history else []
    if recent:
        for i, action_str in enumerate(recent, 1):
            parts.append(f"  {i}. {action_str}")
    else:
        parts.append("  (none yet)")

    parts.append("ACTION:")
    return "\n".join(parts)
