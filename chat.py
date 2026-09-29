"""Raksam AI V3 console."""
from __future__ import annotations
from agent_core import RaksamAgent

def main():
    print("="*72); print("RAKSAM AI — Raksam Brain V3 (RB3)"); print("="*72)
    print("Created by: Prosanto Raksam")
    print("Brain + Eyes + Hands | autonomous observe-think-act-verify")
    print("Commands: /dry-run  /run  /state  /devices  /history  /clear  /quit")
    print("="*72)
    try: agent=RaksamAgent()
    except Exception as e:
        print(f"\nRaksam: Could not load Raksam: {e}"); return
    dry_run=False
    print("\nRaksam: Ready. Autonomous execution is ON.\n")
    while True:
        try:
            user=input("You: ").strip()
            if not user: continue
            cmd=user.lower()
            if cmd in {"/quit","/exit","quit","exit","bye"}: print("\nRaksam: Goodbye."); break
            if cmd=="/dry-run": dry_run=True; print("Raksam: Dry-run ON.\n"); continue
            if cmd=="/run": dry_run=False; print("Raksam: Live execution ON.\n"); continue
            if cmd=="/history":
                print("Raksam memory:"); [print(f"  {t.role}: {t.content}") for t in agent.turns[-12:]]; print(); continue
            if cmd=="/state":
                state=agent.last_state
                print("Raksam state:\n"+(state.to_prompt_text() if state else "No screen state captured yet.")+"\n"); continue
            if cmd=="/devices": print("Connected devices:\n"+agent.devices.describe()+"\n"); continue
            if cmd=="/clear": agent.turns.clear(); agent.history.clear(); agent.last_result=None; print("Raksam: Conversation memory cleared.\n"); continue
            result=agent.chat(user,confirm=False,dry_run=dry_run)
            print(f"\nRaksam: {result.get('response','')}\n")
        except KeyboardInterrupt: print("\nRaksam: Interrupted."); break
        except Exception as e: print(f"\nRaksam: Error — {e}\n")
if __name__=="__main__": main()
