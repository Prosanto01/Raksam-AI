"""Voice-first Raksam console: listen, act, speak, then learn from feedback."""
from __future__ import annotations

from agent_core import RaksamAgent
from voice.windows_voice import listen_once, speak


def _say(text: str) -> None:
    print(f"\nRaksam: {text}\n")
    try: speak(text)
    except Exception as exc: print(f"[Voice output unavailable: {exc}]")


def _listen(prompt: str, timeout: int = 12) -> str:
    _say(prompt)
    try:
        heard = listen_once(timeout)
        print(f"You: {heard}")
        return heard.strip()
    except Exception as exc:
        print(f"[Voice input unavailable: {exc}]")
        return ""


def main() -> None:
    try: agent = RaksamAgent()
    except Exception as exc:
        print(f"Raksam could not start: {exc}"); return
    _say("Voice mode is ready. Tell me one task at a time. Say stop to exit.")
    while True:
        task = _listen("What would you like me to do?", timeout=20)
        if not task:
            _say("I did not hear a task."); continue
        if task.lower() in {"stop", "quit", "exit", "goodbye"}:
            _say("Goodbye."); break
        result = agent.chat(task, dry_run=False)
        _say(result.get("response", "I finished processing the task."))
        if result.get("mode") == "action":
            feedback = _listen("Was the task completed correctly? Say yes, no, or skip.")
            answer = feedback.lower()
            if answer in {"yes", "yeah", "correct", "good"}:
                count = agent.learn_from_last_task(True, "voice confirmation")
                _say(f"Saved {count} approved steps for later training.")
            elif answer in {"no", "wrong", "incorrect"}:
                agent.learn_from_last_task(False, "voice correction")
                _say("Saved the failure. Please demonstrate the correct steps later so I can learn them.")


if __name__ == "__main__": main()
