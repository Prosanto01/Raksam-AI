"""
actuation/permission_gate.py

Every action the brain proposes passes through here BEFORE executor.py
touches the real mouse/keyboard/apps. This is plain rule-based Python -
deliberately NOT a neural net, so its behavior is predictable and
auditable.

Decision values returned by check():
    "allow"   -> execute immediately
    "confirm" -> ask the user Y/N before executing
    "block"   -> never execute, log and skip
"""

from __future__ import annotations

import yaml


class PermissionGate:
    def __init__(self, config_path: str = "config/permissions.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def check(self, action: dict) -> str:
        kind = action.get("action", "")
        target = str(action.get("target", ""))
        device = str(action.get("device", "this_pc"))

        # Only explicitly configured devices may be controlled.
        allowed_devices = [d.lower() for d in self.cfg.get("allowed_devices", ["this_pc"])]
        if device.lower() not in allowed_devices:
            return "block"

        # 1. Hard blocklist always wins, no exceptions.
        if kind in self.cfg.get("hard_blocklist", {}).get("actions", []):
            return "block"
        for blocked_app in self.cfg.get("hard_blocklist", {}).get("apps", []):
            if blocked_app.lower() in target.lower():
                return "block"

        # 2. App allowlist - only enforced for app-touching actions.
        if kind in ("launch_app", "close_app"):
            allowed = [a.lower() for a in self.cfg.get("allowed_apps", [])]
            if target.lower() not in allowed:
                return "block"

        # 3. Per-action rule, else default mode.
        rules = self.cfg.get("action_rules", {})
        return rules.get(kind, self.cfg.get("default_mode", "confirm"))

    def ask_user_confirmation(self, action: dict) -> bool:
        print(f"\n[CONFIRM NEEDED] {action}")
        answer = input("Approve this action? (y/N): ").strip().lower()
        return answer == "y"
