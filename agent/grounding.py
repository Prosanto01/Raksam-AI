"""Ground abstract brain actions onto the current device observation."""
from __future__ import annotations
import re

def _norm(s): return re.sub(r"[^a-z0-9]+","",str(s).lower())

def ground_action(action:dict, observation, goal="") -> dict:
    a=dict(action or {})
    # Brain may return a target but no coordinates. Ground it to UI elements.
    target=str(a.get("target","")).strip()
    kind=str(a.get("action","")).lower()
    if kind in {"click","double_click","move_mouse"} and ("x" not in a or "y" not in a):
        candidates=getattr(observation,"elements",[])
        q=_norm(target)
        scored=[]
        for el in candidates:
            n=_norm(getattr(el,"name",""));
            if not n: continue
            score=0
            if n==q: score=100
            elif q and q in n: score=80
            elif n and n in q: score=60
            if str(getattr(el,"control_type","" )).lower() in {"button","hyperlink","menuitem","tabitem","checkbox","listitem","edit"}: score+=5
            if score: scored.append((score,el))
        if scored:
            _,el=max(scored,key=lambda x:x[0]); c=el.center; a["x"]=c[0]; a["y"]=c[1]; a["grounded_target"]=el.name
    return a
