"""Experience replay store for Raksam's continual learning loop.

Every interaction is recorded as a structured, provenance-aware example.  The
store separates raw experience from *trusted* training examples so a mistake,
malicious web page, or accidental action cannot immediately become training
truth.
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path

ROOT = Path("data/experiences")
RAW = ROOT / "events.jsonl"
APPROVED = ROOT / "approved.jsonl"
REJECTED = ROOT / "rejected.jsonl"


def _write(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def record(kind: str, payload: dict, outcome: str = "unknown", trust: float = 0.2, tags=None) -> dict:
    event = {
        "id": hashlib.sha256(f"{time.time_ns()}:{kind}".encode()).hexdigest()[:16],
        "ts": time.time(), "kind": kind, "outcome": outcome,
        "trust": max(0.0, min(1.0, float(trust))), "tags": tags or [], "payload": payload,
    }
    _write(RAW, event)
    return event


def label(event_id: str, accepted: bool, reason: str = "") -> bool:
    if not RAW.exists(): return False
    found = None
    lines = RAW.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            e = json.loads(line)
            if e.get("id") == event_id:
                found = e; break
        except json.JSONDecodeError:
            continue
    if not found: return False
    found["human_label"] = "accepted" if accepted else "rejected"
    found["label_reason"] = reason
    found["trust"] = 1.0 if accepted else 0.0
    _write(APPROVED if accepted else REJECTED, found)
    return True


def iter_training_events(min_trust: float = 0.75):
    """Yield only high-trust experiences for supervised replay."""
    if not APPROVED.exists(): return
    with APPROVED.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if float(e.get("trust", 0)) >= min_trust:
                    yield e
            except json.JSONDecodeError:
                pass
