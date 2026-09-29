"""Lightweight persistent episodic memory; no embedding model required."""
from __future__ import annotations
import json,re,time
from pathlib import Path
MEM=Path('data/memory/episodic.jsonl')
def toks(s): return set(re.findall(r"[\w']+",s.lower()))
def remember(kind,text,metadata=None):
 MEM.parent.mkdir(parents=True,exist_ok=True)
 with MEM.open('a',encoding='utf8') as f:f.write(json.dumps({'ts':time.time(),'kind':kind,'text':text,'metadata':metadata or {}},ensure_ascii=False)+'\n')
def recall(query,k=5):
 if not MEM.exists():return []
 q=toks(query); scored=[]
 for line in MEM.read_text(encoding='utf8').splitlines()[-5000:]:
  try:r=json.loads(line)
  except:continue
  t=toks(r.get('text','')); score=len(q&t)/(len(q|t) or 1)
  if score>0:scored.append((score,r))
 return [r for _,r in sorted(scored,key=lambda z:z[0],reverse=True)[:k]]
