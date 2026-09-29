"""Raksam V3 Eyes: device/environment perception.

This layer contains no AI model. It gathers facts from the connected device:
- Windows screen/UI tree/OCR
- running windows/processes
- installed Start Menu applications
- optional external camera frames
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os, platform, subprocess, time, json

try:
    from perception.screen_state import perceive
except Exception:
    perceive = None

@dataclass
class DeviceObservation:
    device_id: str
    device_kind: str
    timestamp: float
    state_text: str
    applications: list[dict]
    windows: list[dict]
    metadata: dict

    def to_dict(self):
        return asdict(self)


def _start_menu_apps() -> list[dict]:
    roots = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        roots.append(Path(appdata) / "Microsoft/Windows/Start Menu/Programs")
    if programdata:
        roots.append(Path(programdata) / "Microsoft/Windows/Start Menu/Programs")
    out = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in {".lnk", ".exe", ".appref-ms"}:
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": p.stem, "path": str(p), "type": p.suffix.lower()})
    out.sort(key=lambda x: x["name"].lower())
    return out[:3000]


def _running_windows() -> list[dict]:
    if os.name != "nt":
        return []
    try:
        from pywinauto import Desktop
        result = []
        for w in Desktop(backend="uia").windows():
            try:
                title = (w.window_text() or "").strip()
                if title:
                    r = w.rectangle()
                    result.append({"title": title, "left": r.left, "top": r.top, "right": r.right, "bottom": r.bottom})
            except Exception:
                pass
        return result[:300]
    except Exception:
        return []


def scan_device(device_id: str = "this_pc", kind: str = "pc", include_apps: bool = True) -> DeviceObservation:
    now = time.time()
    state = ""
    if kind == "pc" and perceive is not None:
        try:
            state = perceive().to_prompt_text()
        except Exception as exc:
            state = f"SCREEN_SCAN_ERROR: {exc}"
    apps = _start_menu_apps() if include_apps and os.name == "nt" else []
    windows = _running_windows() if os.name == "nt" else []
    meta = {"platform": platform.platform(), "python": platform.python_version()}
    return DeviceObservation(device_id, kind, now, state, apps, windows, meta)


def observation_prompt(obs: DeviceObservation, max_apps: int = 120) -> str:
    lines = [f"DEVICE: {obs.device_id} ({obs.device_kind})", "OBSERVATION:", obs.state_text[:7000]]
    if obs.applications:
        lines.append("INSTALLED_APPLICATIONS:")
        lines.extend(f"- {a['name']} | {a['path']}" for a in obs.applications[:max_apps])
    if obs.windows:
        lines.append("OPEN_WINDOWS:")
        lines.extend(f"- {w['title']} [{w['left']},{w['top']},{w['right']},{w['bottom']}]" for w in obs.windows[:100])
    return "\n".join(lines)
