"""Raksam autonomous continual-learning orchestrator.

Collects web knowledge plus Raksam's own experiences, builds replay data, and
trains candidates. Human feedback is preferred; automatically successful
low-risk actions may enter replay at high confidence, while raw conversation
outputs remain quarantined until approved or otherwise validated.
"""
from __future__ import annotations
import argparse, json, time, subprocess, sys
from pathlib import Path
from web_learner.continual import run_learning, promote_candidate, CANDIDATE
from web_learner.experience import iter_training_events

CHAT = Path("data/conversations/autonomous_experience.jsonl")
ACTIONS = Path("data/demonstrations/autonomous_experience.jsonl")


def build_replay() -> int:
    CHAT.parent.mkdir(parents=True, exist_ok=True); ACTIONS.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild deterministic replay files from high-trust approved experiences.
    chats, actions = [], []
    for e in iter_training_events():
        p = e.get("payload", {})
        if e.get("kind") == "conversation":
            chats.append({"user": p.get("user", ""), "assistant": p.get("assistant", ""), "history": p.get("history", []), "source": "experience_replay"})
        elif e.get("kind") in {"action", "agent_step"} and p.get("action"):
            prompt=p.get("goal", "")
            if e.get("kind") == "agent_step" and p.get("before_observation"):
                prompt=f"GOAL: {p.get('goal', '')}\nCURRENT_OBSERVATION:\n{p['before_observation']}"
            actions.append({"prompt_text": prompt, "action": p["action"], "device": p["action"].get("device", "this_pc"), "source": "experience_replay", "result": p.get("result")})
    if chats: CHAT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in chats)+"\n", encoding="utf-8")
    if actions: ACTIONS.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in actions)+"\n", encoding="utf-8")
    return len(chats) + len(actions)


def cycle() -> dict:
    replay = build_replay()
    web = run_learning()
    trained = False; log = ""
    # Web learner may have trained already. Run experience replay training when
    # enough trusted examples exist; candidate promotion remains controlled by
    # the web learner's validation policy.
    if replay >= 10:
        cmd = [sys.executable, "brain/train.py", "--candidate", "--resume", "--epochs", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        trained = proc.returncode == 0
        if trained and CANDIDATE.exists():
            try:
                promote_candidate()
                promoted = True
            except Exception as exc:
                promoted = False
                log = (proc.stdout + "\n" + proc.stderr + f"\nPromotion error: {exc}")[-12000:]
            else:
                log = (proc.stdout + "\n" + proc.stderr)[-12000:]
        else:
            promoted = False
        if not log:
            log = (proc.stdout + "\n" + proc.stderr)[-12000:]
    else:
        promoted = False
    return {"replay_examples": replay, "web": web, "experience_training": trained, "promoted": promoted, "training_log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()
    while True:
        try: print(json.dumps(cycle(), ensure_ascii=False, indent=2))
        except Exception as exc: print(json.dumps({"status":"error","error":str(exc)}))
        if args.loop <= 0: break
        time.sleep(args.loop)
