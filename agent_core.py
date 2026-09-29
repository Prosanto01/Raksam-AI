"""Raksam AI V3 — universal Brain + Eyes + Hands agent.

Design:
    User -> Brain -> Hands -> Eyes -> Brain -> ... -> verified goal

The neural brain is Raksam's own from-scratch model. Device adapters are
protocol bridges, not other AIs. The core has no per-action approval gate and
no hard-coded app allowlist. Reliability is provided by observation,
verification, bounded loops, and recovery rather than repeated confirmations.

Public API compatibility is kept at the agent boundary so a website/API can
call the same RaksamAgent later without changing the brain.
"""
from __future__ import annotations
import json,re,time
from dataclasses import dataclass
import yaml
from brain.infer import Brain
from devices.registry import DeviceRegistry
from perception.screen_state import perceive
from agent.loop import AgentLoop
from web_learner.experience import label as label_experience, record as record_experience
from memory import remember,recall

@dataclass
class Turn:
    role:str; content:str; timestamp:float

class ConversationEngine:
    def __init__(self,brain,settings,device_catalog=""):
        self.brain=brain; self.settings=settings.get("conversation",{}); self.device_catalog=device_catalog
        self.brain_max_new_tokens=int(settings.get("brain",{}).get("max_new_tokens",180))
    def reply(self,user_text,history,state_text="",last_result=None):
        context=[]
        if state_text: context.append("Current observation:\n"+state_text[:6000])
        if last_result: context.append("Last result: "+str(last_result))
        if self.device_catalog: context.append("Devices:\n"+self.device_catalog)
        memories=recall(user_text,k=5)
        if memories: context.append("Relevant memory:\n"+"\n".join("- "+m.get("text","")[:700] for m in memories))
        prompt=user_text+("\n\n[CONTEXT]\n"+"\n".join(context) if context else "")
        requested=int(self.settings.get("max_new_tokens",self.brain_max_new_tokens))
        return self.brain.chat(prompt,history[-12:],max_new_tokens=min(requested,self.brain_max_new_tokens))

class RaksamAgent:
    def __init__(self,settings_path="config/settings.yaml"):
        with open(settings_path,encoding="utf-8") as f:self.settings=yaml.safe_load(f) or {}
        b=self.settings.get("brain",{})
        self.brain=Brain(b.get("checkpoint_path","brain/checkpoints/latest.pt"),b.get("tokenizer_path","brain/tokenizer/tokenizer.json"))
        self.devices=DeviceRegistry()
        self.conversation=ConversationEngine(self.brain,self.settings,self.devices.describe())
        self.last_task_event_ids=[]
        self.loop=AgentLoop(self.brain,self.devices,self.settings.get("loop",{}).get("max_steps",25),record=self._record_agent_step)
        self.turns=[]; self.history=[]; self.last_state=None; self.last_action=None; self.last_result=None; self.last_event_id=None

    def _record_agent_step(self,event):
        exp=record_experience("agent_step",event,outcome="success" if event.get("success") else "failure",trust=.75 if event.get("success") else .1,tags=["verified" if event.get("success") else "unverified"])
        self.last_task_event_ids.append(exp["id"]); self.last_event_id=exp["id"]

    def learn_from_last_task(self, accepted: bool, reason: str = "") -> int:
        """Move user-reviewed live steps into the approved/rejected training queue."""
        count=0
        for event_id in self.last_task_event_ids:
            count+=int(label_experience(event_id,accepted,reason))
        return count

    def _remember(self,role,text):
        self.turns.append(Turn(role,str(text),time.time())); self.turns=self.turns[-30:]
    def _remember_action(self,text): self.history.append(str(text)); self.history=self.history[-20:]

    @staticmethod
    def _device_from_text(text):
        low=text.lower(); device="this_pc"
        for marker,candidate in ((" on my phone","my_android"),(" on my android","my_android"),(" on my pc","this_pc"),(" on my computer","this_pc")):
            if marker in low:
                text=text[:low.rfind(marker)].strip(); return text,candidate
        return text,device

    @staticmethod
    def _looks_like_action(text):
        t=text.lower().strip()
        verbs=("open ","launch ","start ","close ","click ","double click ","type ","write ","press ","scroll ","move mouse","go to ","select ","paste ","copy ","save ","search ","find ","tap ","swipe ","play ","send ","download ","upload ")
        return t.startswith(verbs) or t.startswith(("in ","inside ")) or any(p in t for p in ("open the","click the","search for","search on","play "," on my phone"," on my computer"))

    def _snapshot(self,device_id="this_pc"):
        try:
            if device_id=="this_pc": self.last_state=perceive(); return self.last_state.to_prompt_text()
            return self.devices.get(device_id).observe()
        except Exception as e:return f"OBSERVATION_ERROR: {e}"

    def _execute_one(self,goal,action):
        device_id=action.get("device","this_pc")
        adapter=self.devices.get(device_id)
        before_obj=self.last_state if device_id=="this_pc" else None
        try:
            result=adapter.execute(action)
        except Exception as e: result=f"ERROR: {e}"
        after_text=self._snapshot(device_id)
        success="error" not in str(result).lower() and "not found" not in str(result).lower()
        if action.get("action")=="launch_app" and device_id=="this_pc":
            target=re.sub(r"[^a-z0-9]+","",str(action.get("target","" )).lower()); current=re.sub(r"[^a-z0-9]+","",after_text.lower()); success=success and (target in current or any(w in current for w in target.split() if len(w)>2) or str(action.get("target","" )).strip().lower() in str(self.last_result or "").lower())
        event={"goal":goal,"device":device_id,"action":action,"result":result,"success":success,"observation":after_text[:3000]}
        self.last_action=action; self.last_result=result
        exp=record_experience("action",event,outcome="success" if success else "failure",trust=.85 if success else .15); self.last_event_id=exp["id"]
        remember("action",json.dumps(event,ensure_ascii=False))
        self._remember_action(json.dumps(event,ensure_ascii=False))
        return event

    def chat(self,user_text,confirm=False,dry_run=False):
        self._remember("user",user_text)
        text,device=self._device_from_text(user_text)
        if self._looks_like_action(text):
            # Generic Brain-driven agent loop: observe -> think -> act -> observe -> verify/replan.
            if dry_run:
                obs=self._snapshot(device); response="Raksam plan (dry-run):\n"+obs[:2500]; return {"mode":"action","goal":user_text,"response":response,"success":True,"observation":obs}
            self.last_task_event_ids=[]
            events=self.loop.run(user_text,device_id=device)
            ok=bool(events) and bool(events[-1].get("success")) and events[-1].get("action",{}).get("action")=="done"
            response="Task completed." if ok else "I worked through the available observations, but the task was not verified as complete."
            self._remember("assistant",response); return {"mode":"action","goal":user_text,"response":response,"success":ok,"events":events}

        state=""
        if any(x in text.lower() for x in ("screen","window","what do you see","what's open")): state=self._snapshot(device)
        response=self.conversation.reply(text,self.turns[:-1],state,self.last_result)
        self._remember("assistant",response); remember("conversation",f"User: {text}\nAssistant: {response}")
        exp=record_experience("conversation",{"user":text,"assistant":response},outcome="generated",trust=.2); self.last_event_id=exp["id"]
        return {"mode":"conversation","response":response,"experience_id":exp["id"]}

    def step(self,goal,dry_run=False,confirm=False,on_event=None):
        result=self.chat(goal,confirm=confirm,dry_run=dry_run)
        if on_event:on_event(result)
        return result
    def run(self,goal,dry_run=False,confirm=False,max_steps=None,on_event=None):
        return self.loop.run(goal,max_steps=max_steps or self.settings.get("loop",{}).get("max_steps",25),on_event=on_event)
