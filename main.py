"""Raksam AI task runner.

Uses the same unified RaksamAgent as chat.py, so task execution and chat
share memory, perception, brain inference, permissions, and results.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from agent_core import RaksamAgent


def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raksam")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = Path(log_dir) / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def run_task(goal: str, dry_run: bool):
    agent = RaksamAgent()
    logger = setup_logging(agent.settings["logging"]["log_dir"])

    def on_event(event: dict):
        logger.info(json.dumps(event, ensure_ascii=False))
        action = event["action"]
        prefix = "[DRY RUN]" if dry_run else "[STEP]"
        print(f"{prefix} {action.get('action')} -> {action.get('target', '')}")
        if event.get("result"):
            print(f"        {event['result']}")

    # Simple compound commands are handled deterministically as a sequence.
    explicit_actions = agent._explicit_actions(goal)
    if explicit_actions and len(explicit_actions) > 1:
        blocked = next((a for a in explicit_actions if agent.gate.check(a) == "block"), None)
        if blocked:
            event = agent._execute_planned(goal, blocked, dry_run=True, confirm=False)
            on_event(event)
            events = [event]
        else:
            events = []
            for action in explicit_actions:
                event = agent._execute_planned(goal, action, dry_run=dry_run, confirm=True)
                events.append(event)
                on_event(event)
                if event.get("gate_decision") in {"block", "stopped", "skipped"}:
                    break
    else:
        events = agent.run(
            goal,
            dry_run=dry_run,
            confirm=True,
            max_steps=agent.settings["loop"]["max_steps"],
            on_event=on_event,
        )
    if events and events[-1]["action"].get("action") == "done":
        print("Brain signaled task complete.")
    elif len(events) >= agent.settings["loop"]["max_steps"]:
        print(f"Reached max_steps ({agent.settings['loop']['max_steps']}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Raksam PC agent on one task.")
    parser.add_argument("--goal", type=str, default=None, help="Task goal. Prompts if omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, execute nothing.")
    args = parser.parse_args()
    goal = args.goal or input("What should the agent do?: ").strip()
    run_task(goal, dry_run=args.dry_run)
