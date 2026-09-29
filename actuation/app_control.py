"""Windows application discovery and control for Raksam's Hands.

Raksam does not keep a hard-coded list of applications. It searches the
machine's Start Menu shortcuts and known executable locations, then launches
the best match. This is deterministic infrastructure, not another AI.
"""
from __future__ import annotations
import os, re, subprocess, time, shutil, difflib
from pathlib import Path
import psutil
try:
    from pywinauto import Desktop
except Exception:
    Desktop=None

def discover_apps(limit=3000):
    roots=[]
    for env in ("APPDATA","PROGRAMDATA"):
        value=os.environ.get(env)
        if value: roots.append(Path(value)/"Microsoft/Windows/Start Menu/Programs")
    apps=[]; seen=set()
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in {".lnk",".exe",".appref-ms"}: continue
            key=str(p).lower()
            if key in seen: continue
            seen.add(key); apps.append({"name":p.stem,"path":str(p),"type":p.suffix.lower()})
            if len(apps)>=limit: return apps
    return apps

def _norm(s): return re.sub(r"[^a-z0-9]+","",str(s).lower())

def find_app(name):
    q=_norm(name)
    if not q: return None
    apps=discover_apps()
    exact=[a for a in apps if _norm(a["name"])==q]
    if exact: return exact[0]
    contains=[a for a in apps if q in _norm(a["name"]) or _norm(a["name"]) in q]
    if contains: return sorted(contains,key=lambda a:abs(len(_norm(a["name"]))-len(q)))[0]
    # Small typo-tolerance: compare against discovered applications.
    scored=[]
    for a in apps:
        n=_norm(a["name"])
        if not n: continue
        score=difflib.SequenceMatcher(None,q,n).ratio()
        if score >= 0.70: scored.append((score,a))
    if scored:
        scored.sort(key=lambda item:item[0], reverse=True)
        return scored[0][1]
    return None

def launch_app(name_or_path, wait_seconds=1.0):
    target=str(name_or_path).strip()
    found=find_app(target) if not os.path.exists(target) else None
    launch_target=found["path"] if found else target

    # Never hand an unresolved natural-language app name to a shell.
    # Try a real executable on PATH, otherwise report that the app was not found.
    if not found and not os.path.exists(launch_target):
        executable=shutil.which(launch_target) or shutil.which(launch_target + ".exe")
        if executable:
            launch_target=executable
        elif os.name == "nt" and launch_target.lower() in {"notepad","notepad.exe"}:
            launch_target=shutil.which("notepad.exe") or "notepad.exe"
        else:
            raise FileNotFoundError(f"Application not found: {target}")

    if os.name != "nt":
        subprocess.Popen([launch_target])
    elif launch_target.lower().endswith((".lnk", ".appref-ms")):
        os.startfile(launch_target)
    else:
        subprocess.Popen([launch_target], shell=False)
    time.sleep(max(0,float(wait_seconds)))
    return found or {"name":Path(launch_target).stem,"path":launch_target}

def close_app(process_name):
    q=_norm(process_name); found=False
    for proc in psutil.process_iter(["name"]):
        try:
            n=proc.info.get("name") or ""
            if _norm(n)==q or q in _norm(n): proc.terminate(); found=True
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return found

def focus_window(title_substring):
    if Desktop is None:
        return False
    q=str(title_substring or "").strip().lower()
    if not q:
        return False
    try:
        desktop=Desktop(backend="uia")
        candidates=[]
        for w in desktop.windows():
            try:
                title=(w.window_text() or "").strip()
                if q in title.lower():
                    candidates.append(w)
            except Exception:
                pass
        if not candidates:
            return False
        # Prefer the last matching top-level window, which is usually the
        # newly launched foreground application.
        w=candidates[-1]
        try: w.restore()
        except Exception: pass
        w.set_focus()
        return True
    except Exception:
        return False

def is_running(process_name):
    q=_norm(process_name)
    return any(_norm((p.info.get("name") or ""))==q for p in psutil.process_iter(["name"]))
