"""Goal semantics for the RB3 control loop.

This is deliberately a small, app-agnostic parser.  It only preserves facts
the user stated (resource, requested UI verb, and text), leaving grounding and
all unfamiliar next steps to the RB3 brain after each fresh observation.
"""
from __future__ import annotations

import re


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" .")


def _poem(topic: str) -> str:
    """A dependable offline fallback when RB3 cannot supply generated prose."""
    subject = _clean(topic) or "the world"
    return "\n".join((
        f"Moonlight gathers over {subject},",
        "A silver hush along the sky,",
        "Night carries its quiet promise,",
        "And stars keep watch as dreams pass by.",
    ))


class TaskSemantics:
    """Extract a bounded, ordered scaffold for a natural-language task."""
    def __init__(self, goal: str, device: str = "this_pc", composer=None):
        self.goal = _clean(goal)
        self.device = device
        self.composer = composer
        self.resource = ""
        self.text = ""
        self.needs_compose = False
        self._stage = 0
        self._parse()

    def _parse(self):
        # Split only on command connectors.  This avoids treating words in the
        # requested text itself as separate actions.
        opened = re.match(r"^(?:open|launch|start)\s+(?:the\s+)?(.+?)(?:\s+(?:and|then)\s+)(.+)$", self.goal, re.I)
        remainder = self.goal
        if opened:
            self.resource, remainder = _clean(opened.group(1)), _clean(opened.group(2))
        else:
            simple_open = re.match(r"^(?:open|launch|start)\s+(?:the\s+)?(.+)$", self.goal, re.I)
            if simple_open:
                self.resource = _clean(simple_open.group(1))

        self.needs_compose = bool(re.search(r"\b(?:compose|new message|email)\b", remainder, re.I))
        poem = re.search(r"\b(?:write|create|compose)\s+(?:a\s+)?poem\s+(?:about|on)\s+(.+)$", remainder, re.I)
        if poem:
            request = f"Write a short four-line poem about {_clean(poem.group(1))}. Return only the poem."
            candidate = self.composer(request) if self.composer else ""
            # Model output is optional: never type a planning/error message.
            self.text = _clean(candidate) if candidate and len(_clean(candidate)) > 12 else _poem(poem.group(1))
            return
        saying = re.search(r"\b(?:message|email)\s+(?:saying|that says)\s+(.+)$", remainder, re.I)
        direct = re.search(r"\b(?:write|type)\s+(.+)$", remainder, re.I)
        self.text = _clean((saying or direct).group(1)) if (saying or direct) else ""

    @property
    def is_compound(self) -> bool:
        # Opening a resource is itself a complete, scaffoldable task.  Without
        # this, a simple request such as "open Notepad" falls through to the
        # neural planner after launch and can repeatedly open the same app.
        return bool(self.resource)

    def next_action(self, observation: str, previous_action: dict | None = None) -> dict | None:
        """Return only the next safe scaffold action; never bulk-execute a plan."""
        if not self.is_compound:
            return None
        # Do not march on after a failed open/click.  Yield to RB3 with the
        # failure and newest observation so it can choose a recovery action.
        if previous_action and not previous_action.get("success", True):
            return None
        if self._stage == 0:
            self._stage = 1
            # The executor discovers local applications first, then treats an
            # unresolved single resource name as a web destination.  No named
            # service list belongs here.
            return {"action": "open_resource", "target": self.resource,
                    "device": self.device, "reason": "open requested resource before interacting"}
        if self._stage == 1 and self.needs_compose:
            self._stage = 2
            return {"action": "click", "target": "compose", "device": self.device,
                    "reason": "user requested a new message; ground this label in the current UI"}
        if self._stage <= 2 and self.text:
            self._stage = 3
            return {"action": "type", "target": self.text, "device": self.device,
                    "reason": "type requested content after fresh UI observation"}
        if self._stage < 4:
            self._stage = 4
            return {"action": "done", "target": "", "device": self.device,
                    "reason": "all requested scaffold steps were verified"}
        return None
