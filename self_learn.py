"""Autonomous Raksam learning daemon.

Raksam can continuously collect owner-approved public data, train its own
transformer, and (when enabled) promote a validated candidate automatically.
No external AI model is used.
"""
from __future__ import annotations
import argparse, json, time
from web_learner.continual import run_learning

p = argparse.ArgumentParser()
p.add_argument('--once', action='store_true')
p.add_argument('--loop', type=int, default=0, help='repeat every N seconds')
args = p.parse_args()

def one(): print(json.dumps(run_learning(), ensure_ascii=False, indent=2))

if args.loop > 0:
    while True:
        try: one()
        except KeyboardInterrupt: break
        except Exception as exc: print(json.dumps({'status':'error','error':str(exc)}))
        time.sleep(args.loop)
else:
    one()
