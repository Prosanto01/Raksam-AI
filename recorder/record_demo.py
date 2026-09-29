"""
recorder/record_demo.py

BUILD/RUN THIS FIRST. Without demonstration data there is nothing to
train the brain on.

How it works:
  1. You tell it what GOAL you're about to demonstrate (e.g. "open
     notepad and type hello").
  2. It watches your mouse clicks and keystrokes.
  3. Before each of YOUR actions, it captures the current screen state
     (via perception/screen_state.py).
  4. After you act, it records what you did, paired with the state
     that came before it.
  5. Press F9 to end the current demonstration (marks it "done").

Each demonstration is saved as one JSONL file in data/demonstrations/,
one line per (state, action) step - exactly the shape brain/dataset.py
expects to train on.

Run:
    python recorder/record_demo.py
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import keyboard
import mouse  # part of the `mouse` package if available; falls back below
import yaml

from perception.screen_state import perceive
from perception.serializer import build_prompt

OUTPUT_DIR = Path("data/demonstrations")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DemoRecorder:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:8]
        self.steps: list[dict] = []
        self.history: list[str] = []
        self.goal = ""
        self.stop_flag = False

    def start(self):
        self.goal = input("What task are you about to demonstrate? (GOAL): ").strip()
        print(f"\nRecording started. Session id: {self.session_id}")
        print("Perform the task normally. Press F9, F10, ESC, or End when you're done with this demo.\n")

        # Add multiple hotkeys to finish recording
        keyboard.add_hotkey("f9", self._on_stop)
        keyboard.add_hotkey("f10", self._on_stop)
        keyboard.add_hotkey("esc", self._on_stop)
        keyboard.add_hotkey("end", self._on_stop)

        # Record on every mouse click and every key press.
        # (Using the `keyboard`/`mouse` libraries' hook APIs.)
        keyboard.hook(self._on_key_event)
        mouse.hook(self._on_mouse_event)

        while not self.stop_flag:
            time.sleep(0.1)

        self._save()

    def _on_stop(self):
        self.stop_flag = True

    def _capture_pre_action_state(self):
        shot_path = str(SCREENSHOT_DIR / f"{self.session_id}_{len(self.steps):03d}.png")
        state = perceive(save_screenshot_to=shot_path)
        prompt_text = build_prompt(self.goal, state, self.history)
        return state, prompt_text

    def _record_step(self, action: dict, action_str: str):
        state, prompt_text = self._capture_pre_action_state()
        self.steps.append(
            {
                "goal": self.goal,
                "prompt_text": prompt_text,
                "window_title": state.active_window_title,
                "action": action,
            }
        )
        self.history.append(action_str)
        print(f"  recorded: {action_str}")

    def _on_mouse_event(self, event):
        # `mouse` package fires ButtonEvent on click with event_type "down"/"up"
        if getattr(event, "event_type", None) == "down":
            x, y = mouse.get_position()
            action = {"action": "click", "target": "", "x": x, "y": y, "reason": "demonstrated"}
            self._record_step(action, f'click({x},{y})')

    def _on_key_event(self, event):
        if event.event_type != "down":
            return
        # Don't record the stop hotkeys themselves
        if event.name in ["f9", "f10", "esc", "end"]:
            return
        action = {"action": "key", "target": event.name, "reason": "demonstrated"}
        self._record_step(action, f'key("{event.name}")')

    def _save(self):
        out_path = OUTPUT_DIR / f"demo_{self.session_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for step in self.steps:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")
            # Final "done" step so the brain learns when to stop.
            f.write(json.dumps({
                "goal": self.goal,
                "prompt_text": "",
                "window_title": "",
                "action": {"action": "done", "target": "", "reason": "task finished"},
            }, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(self.steps)} steps to {out_path}")


if __name__ == "__main__":
    recorder = DemoRecorder()
    recorder.start()
