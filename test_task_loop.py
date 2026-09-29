"""Offline regression checks for generic, multi-step task handling."""
from agent.semantics import TaskSemantics
from agent.loop import AgentLoop


def test_semantics_preserves_literal_and_generated_text():
    literal = TaskSemantics("Open Notepad and write hello, world")
    assert literal.next_action("")["action"] == "open_resource"
    assert literal.next_action("", {"success": True})["action"] == "type"
    poem = TaskSemantics("Open Notepad and write a poem about the moon", composer=lambda _: "")
    poem.next_action("")
    typed = poem.next_action("", {"success": True})["target"]
    assert "Moonlight" in typed and "moon" in typed.lower()


def test_compose_is_grounded_before_typing():
    task = TaskSemantics("Open Gmail and compose a message saying hello there")
    assert task.next_action("")["action"] == "open_resource"
    assert task.next_action("", {"success": True}) == {"action": "click", "target": "compose", "device": "this_pc", "reason": "user requested a new message; ground this label in the current UI"}
    assert task.next_action("", {"success": True})["target"] == "hello there"


def test_failure_does_not_skip_to_typing():
    task = TaskSemantics("Open Notepad and write keep me safe")
    task.next_action("")
    assert task.next_action("", {"success": False}) is None


def test_simple_open_finishes_after_one_launch():
    task = TaskSemantics("Open Notepad")
    assert task.next_action("")["action"] == "open_resource"
    assert task.next_action("", {"success": True})["action"] == "done"


class FakeAdapter:
    def __init__(self): self.actions = []
    def observe(self): return "WINDOW: test"
    def execute(self, action): self.actions.append(action); return "ok"

class FakeDevices:
    def __init__(self): self.adapter = FakeAdapter()
    def get(self, _): return self.adapter

class FakeBrain:
    def chat(self, *_args, **_kwargs): return ""
    def act(self, *_args, **_kwargs): return {"action": "done", "target": ""}

def test_loop_observes_between_scaffold_actions():
    devices = FakeDevices()
    loop = AgentLoop(FakeBrain(), devices, max_steps=3)
    events = loop.run("Open Notepad and write checked", device_id="fake")
    assert [e["action"]["action"] for e in events[:2]] == ["open_resource", "type"]
    assert len(events) == 3  # RB3 receives a subsequent observation and decides completion.


if __name__ == "__main__":
    test_semantics_preserves_literal_and_generated_text()
    test_compose_is_grounded_before_typing()
    test_failure_does_not_skip_to_typing()
    test_simple_open_finishes_after_one_launch()
    test_loop_observes_between_scaffold_actions()
    print("PASS: task semantics and observe-think-act-verify loop")
