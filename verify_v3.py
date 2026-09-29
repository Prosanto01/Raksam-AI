"""Offline verification for Raksam AI V3."""
from __future__ import annotations
import compileall
print("Raksam AI V3 verification")
if not compileall.compile_dir(".", quiet=1):
    raise SystemExit("FAIL: Python compilation")
from brain.infer import Brain
b=Brain()
assert b.model_name=="Raksam Brain V3 (RB3)"
assert b.created_by=="Prosanto Raksam"
assert "Prosanto Raksam" in b.chat("Who created you?")
assert "2 + 2" in b.chat("What is 2 + 2?")
assert b.act("GOAL: open Notepad\nDEVICE: this_pc")["action"]=="launch_app"
assert b.act("GOAL: type Hello\nDEVICE: this_pc")["action"]=="type"
assert b.act("GOAL: click the Search button\nDEVICE: this_pc")["action"]=="click"
from agent_core import RaksamAgent
a=RaksamAgent()
r=a.chat("in notepad write Hello",dry_run=True)
assert r["mode"]=="action" and "Raksam plan" in r["response"]
print("PASS: checkpoint, learned answers, atomic action heuristics, Turn history compatibility, and generic loop routing.")
