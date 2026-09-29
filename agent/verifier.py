"""Post-action verification using fresh observations."""
from __future__ import annotations
import re

def _norm(s): return re.sub(r"\s+"," ",str(s or "").strip().lower())

def _state_text(state):
    if hasattr(state, "to_prompt_text"):
        try: return state.to_prompt_text()
        except Exception: pass
    return str(getattr(state, "state_text", state) or "")


def _fingerprint(state):
    """Small stable view of observable UI state for post-action checks."""
    title = str(getattr(state, "active_window_title", "") or "")
    elements = tuple(
        (str(getattr(el, "control_type", "")), str(getattr(el, "name", "")),
         int(getattr(el, "left", 0)), int(getattr(el, "top", 0)),
         int(getattr(el, "right", 0)), int(getattr(el, "bottom", 0)))
        for el in getattr(state, "elements", [])
    )
    return title, elements, _state_text(state)


def _changed(before, after) -> bool:
    return _fingerprint(before) != _fingerprint(after)

def verify(action, before, after, result_text=""):
    kind=str(action.get("action","")).lower()
    target=str(action.get("target","")).strip()
    result=_norm(result_text)
    if "error" in result or "not found" in result: return False, "executor reported failure"
    if kind=="done": return False,"completion needs a verified prior action"
    if kind=="launch_app":
        q=_norm(target)
        title=_norm(getattr(after,"active_window_title",""))
        state=_norm(_state_text(after))
        ok=bool(q) and (q in title or q in state)
        return ok, "application/window observed" if ok else "application not visible yet"
    if kind == "open_resource":
        # The local app launcher has already waited and focused the selected
        # application.  Treat its explicit result as sufficient for this one
        # step, because some native windows expose an unchanged/empty UIA tree
        # immediately after launch and would otherwise be launched repeatedly.
        if "opened application:" in result or "opened web resource:" in result:
            return True, "resource launcher confirmed an open request"
        if _changed(before, after):
            return True, "visible UI state changed after opening resource"
        return False, "resource did not open or become observable"
    if kind in {"open_url", "navigate", "focus_window", "click", "double_click", "scroll", "key", "press"}:
        if _changed(before, after):
            return True, "visible UI state changed after action"
        return False, "no visible UI change after action"
    if kind in {"move_mouse", "wait"}:
        return True, "action executed; no visual state change required"
    if kind in {"type","write"}:
        q=_norm(target)
        text=_norm(_state_text(after))
        if q and q in text: return True,"typed text is visible in UI state"
        if _changed(before, after):
            return True, "visible UI state changed after typing"
        return False, "typed text was not observable after action"
    return False, "unsupported action cannot be verified"
