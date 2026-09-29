"""Raksam V3 Hands executor.

The brain chooses abstract actions. This module performs them through the
ready-made low-level mouse/keyboard/device libraries and returns observations.
No approval gate is used here; the agent loop handles reliability/verification.
"""
from __future__ import annotations
import re, time, webbrowser
from actuation import app_control, mouse_keyboard

def _require_xy(a):
    if "x" not in a or "y" not in a: raise ValueError("click action requires x and y")
    return int(a["x"]),int(a["y"])

def execute(action:dict)->str:
    kind=str(action.get("action","")).lower(); target=action.get("target","")
    if kind in {"click","double_click","move_mouse"}:
        x,y=_require_xy(action)
        if kind=="click": mouse_keyboard.click(x,y,action.get("button","left")); return f"clicked ({x},{y}) target={target}"
        if kind=="double_click": mouse_keyboard.double_click(x,y); return f"double-clicked ({x},{y}) target={target}"
        mouse_keyboard.move_mouse(x,y); return f"moved mouse to ({x},{y})"
    if kind in {"type","write"}:
        mouse_keyboard.focus_active_text_control(click=True)
        mouse_keyboard.type_text(str(target))
        return f"typed: {target!r}"
    if kind in {"key","press"}: mouse_keyboard.press_key(str(target)); return f"pressed key: {target}"
    if kind=="scroll":
        n=int(action.get("amount",target or 0)); mouse_keyboard.scroll(n); return f"scrolled: {n}"
    if kind in {"open_url","navigate"}:
        url=str(target); url=url if url.startswith(("http://","https://")) else "https://"+url
        webbrowser.open(url); return f"opened URL: {url}"
    if kind=="open_resource":
        # Discover installed applications dynamically.  If the requested
        # single-name resource is not installed, use its ordinary web host.
        # This is intentionally not a list of favoured applications.
        resource=str(target).strip()
        found=app_control.find_app(resource)
        if found:
            app_control.launch_app(found["path"], wait_seconds=1.5)
            app_control.focus_window(found["name"])
            return f"opened application: {found['name']}"
        host=re.sub(r"\s+", "", resource).lower()
        if not host or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*", host):
            raise ValueError("resource is neither a discovered application nor a safe web host")
        url=host if host.startswith(("http://","https://")) else "https://" + (host if "." in host else host+".com")
        webbrowser.open(url)
        return f"opened web resource: {url}"
    if kind=="launch_app":
        found=app_control.launch_app(str(target), wait_seconds=1.5)
        app_control.focus_window(str(found.get("name",target)))
        time.sleep(0.35)
        return f"launched app: {found.get('name',target)}"
    if kind=="close_app": return f"closed app: {target}" if app_control.close_app(str(target)) else f"app not running: {target}"
    if kind=="focus_window": return "focused window" if app_control.focus_window(str(target)) else "window not found"
    if kind=="wait":
        s=max(0,float(action.get("seconds",target or 1))); time.sleep(s); return f"waited {s}s"
    if kind=="done": return "task complete"
    raise ValueError(f"Unknown action: {kind}")
