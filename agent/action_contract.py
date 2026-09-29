"""Strict action contract between the brain and device bodies.

The model may suggest actions, but only well-formed actions reach a device.
HTTP bodies remain extensible because they validate their own device protocol.
"""
from __future__ import annotations

from typing import Any


PC_ACTIONS = {
    "click", "double_click", "move_mouse", "type", "key", "scroll",
    "open_url", "navigate", "open_resource", "launch_app", "close_app",
    "focus_window", "wait", "done",
}
ANDROID_ACTIONS = {
    "android_tap", "tap", "android_swipe", "swipe", "android_text",
    "type", "android_key", "key", "android_launch", "launch_app", "done",
}
ALIASES = {"write": "type", "press": "key"}


def validate_action(action: Any, device_kind: str) -> tuple[dict | None, str]:
    """Return a safe, normalized action or a reason it must not execute."""
    if not isinstance(action, dict):
        return None, "brain did not return an action object"
    result = dict(action)
    kind = ALIASES.get(str(result.get("action", "")).strip().lower(), str(result.get("action", "")).strip().lower())
    result["action"] = kind
    target = result.get("target", "")
    if target is None:
        target = ""
    result["target"] = str(target).strip()

    if device_kind == "http":
        # A user-owned bridge is the authority for its custom action schema.
        return result, ""
    allowed = ANDROID_ACTIONS if device_kind == "android_adb" else PC_ACTIONS
    if kind not in allowed:
        return None, f"unsupported {device_kind} action: {kind or 'missing'}"
    if kind in {"type", "key", "open_url", "navigate", "open_resource", "launch_app", "close_app", "focus_window"} and not result["target"]:
        return None, f"{kind} requires a non-empty target"
    if kind in {"click", "double_click", "move_mouse", "tap", "android_tap"}:
        has_coordinates = "x" in result and "y" in result
        if not has_coordinates and not result["target"]:
            return None, f"{kind} requires coordinates or a visible target"
        if has_coordinates:
            try:
                result["x"] = int(result["x"]); result["y"] = int(result["y"])
            except (TypeError, ValueError):
                return None, f"{kind} coordinates must be integers"
    if kind in {"swipe", "android_swipe"}:
        try:
            for field in ("x1", "y1", "x2", "y2"):
                result[field] = int(result[field])
        except (KeyError, TypeError, ValueError):
            return None, "swipe requires integer x1, y1, x2, and y2 coordinates"
    return result, ""
