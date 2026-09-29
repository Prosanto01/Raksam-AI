"""Universal observe-think-act-verify loop for Raksam V3."""
from __future__ import annotations
import time
from .grounding import ground_action
from .verifier import verify
from .semantics import TaskSemantics
from .action_contract import validate_action

class AgentLoop:
    def __init__(self, brain, devices, max_steps=25, record=None):
        self.brain=brain; self.devices=devices; self.max_steps=max_steps; self.record=record
    def run(self, goal, device_id="this_pc", max_steps=None, on_event=None):
        limit=max_steps or self.max_steps; events=[]; failed_actions={}
        history=[]
        semantics=TaskSemantics(goal, device_id, composer=lambda prompt: self.brain.chat(prompt, [], max_new_tokens=100))
        for step in range(limit):
            adapter=self.devices.get(device_id)
            try:
                before=adapter.observe()
            except Exception as exc:
                events.append({"step":step+1,"success":False,"reason":f"observation failed: {exc}"})
                break
            # For PC, registry adapters return serialized observations. Brain gets it directly.
            prompt=(f"GOAL: {goal}\nDEVICE: {device_id}\nCURRENT_OBSERVATION:\n{before[:12000]}\n"
                    f"PREVIOUS_ACTIONS:\n"+"\n".join(history[-8:])+"\n"
                    "Choose ONE next action. Return JSON with action,target and optional x,y/reason. "
                    "Use action='done' only when the goal is visibly achieved.")
            # Compound tasks receive one semantic scaffold action at a time.
            # Every action still observes, grounds, verifies, and can replan.
            action=semantics.next_action(before, events[-1] if events else None)
            if action is None:
                action=self.brain.act(prompt,max_new_tokens=140)
            if not isinstance(action, dict):
                events.append({"step":step+1,"action":action,"success":False,"reason":"brain did not produce an action object"})
                break
            action.setdefault("device",device_id)
            action,error=validate_action(action, getattr(getattr(adapter,"device",None),"kind","pc"))
            if error:
                event={"step":step+1,"action":action,"success":False,"reason":error}
                events.append(event); break
            # Grounding needs structured UI elements. Obtain the richer PC state when possible.
            observation_obj=None
            if device_id=="this_pc":
                try:
                    from perception.screen_state import perceive
                    observation_obj=perceive()
                except Exception: observation_obj=None
            if observation_obj is not None: action=ground_action(action,observation_obj,goal)
            action,error=validate_action(action, getattr(getattr(adapter,"device",None),"kind","pc"))
            if error:
                events.append({"step":step+1,"action":action,"success":False,"reason":error})
                break
            action_key=repr(sorted(action.items()))
            if failed_actions.get(action_key,0)>=2:
                events.append({"step":step+1,"action":action,"success":False,"reason":"same unverified action was attempted twice; stopping for recovery"})
                break
            if action.get("action")=="done" and not any(e.get("success") and e.get("action",{}).get("action")!="done" for e in events):
                events.append({"step":step+1,"action":action,"success":False,"reason":"brain claimed completion before any verified work"})
                break
            try:
                result=adapter.execute(action)
            except Exception as exc:
                result=f"ERROR: {exc}"
            try:
                after_text=adapter.observe()
            except Exception as exc:
                after_text=f"OBSERVATION_ERROR: {exc}"
            after_obj=None
            if device_id=="this_pc":
                try:
                    from perception.screen_state import perceive
                    after_obj=perceive()
                except Exception: after_obj=None
            if action.get("action")=="done":
                # Completion is allowed only after the guard above established
                # that at least one real action was visibly verified.
                success=True; reason="brain completed task after verified work"
            elif after_obj is not None and observation_obj is not None:
                success,reason=verify(action,observation_obj,after_obj,result)
            else:
                success="error" not in str(result).lower(); reason="adapter result without visual verification"
            event={"step":step+1,"goal":goal,"device":device_id,"action":action,"result":result,"success":success,"verification":reason,"before_observation":before[:2000],"observation":after_text[:2000]}
            events.append(event)
            if self.record:
                try: self.record(event)
                except Exception: pass
            if on_event: on_event(event)
            history.append(f"{action} => {result} => verified={success}")
            if action.get("action")=="done" and success: break
            if not success:
                failed_actions[action_key]=failed_actions.get(action_key,0)+1
                # Replanning happens automatically on the next iteration.
                continue
        return events
