"""Basic regression evaluation for active/candidate Raksam brains."""
from __future__ import annotations
import json, pathlib, sys
from brain.infer import Brain
TEST=pathlib.Path("data/evaluation/benchmarks.jsonl")
def run(path):
    b=Brain(str(path)); rows=[]
    for line in TEST.read_text(encoding="utf8").splitlines() if TEST.exists() else []:
        try: r=json.loads(line)
        except: continue
        if r.get("type","chat")!="chat": continue
        out=b.chat(r.get("user",""),[],80)
        rows.append({"user":r.get("user",""),"output":out,"nonempty":bool(out.strip())})
    score=sum(x["nonempty"] for x in rows)/max(1,len(rows))
    return score,rows
if __name__=="__main__":
    for name in sys.argv[1:] or ["brain/checkpoints/latest.pt"]:
        score,rows=run(name); print(f"{name}: reliability={score:.2%}, tests={len(rows)}")
