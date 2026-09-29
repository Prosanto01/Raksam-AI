"""Android ADB device body."""
from __future__ import annotations
import subprocess
from devices.base import Device,DeviceAdapter
class AndroidADBAdapter(DeviceAdapter):
    def __init__(self,device): super().__init__(device); self.serial=device.metadata.get("serial"); self.adb=device.metadata.get("adb_path","adb")
    def _run(self,*args):
        cmd=[self.adb]+(["-s",self.serial] if self.serial else [])+list(args)
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=False)
        if p.returncode: raise RuntimeError(p.stderr.strip() or f"adb failed: {p.returncode}")
        return p.stdout.strip()
    def observe(self):
        size=self._run("shell","wm","size")
        focus=self._run("shell","dumpsys","window","windows")
        focus_lines=[x.strip() for x in focus.splitlines() if "mCurrentFocus" in x or "mFocusedApp" in x]
        return f"DEVICE: {self.device.name} (Android)\nSCREEN: {size}\nFOCUS: {focus_lines[-1] if focus_lines else 'unknown'}\nCAPABILITIES: {', '.join(self.device.capabilities)}"
    def execute(self,a):
        k=a.get("action"); t=str(a.get("target",""))
        if k in {"android_tap","tap"}: return self._run("shell","input","tap",str(a["x"]),str(a["y"])) or f"tapped {a['x']},{a['y']}"
        if k in {"android_swipe","swipe"}: return self._run("shell","input","swipe",str(a["x1"]),str(a["y1"]),str(a["x2"]),str(a["y2"]),str(a.get("duration_ms",300))) or "swiped"
        if k in {"android_text","type"}: return self._run("shell","input","text",t.replace(" ","%s")) or "typed text"
        if k in {"android_key","key"}: return self._run("shell","input","keyevent",t) or f"sent keyevent {t}"
        if k in {"android_launch","launch_app"}: return self._run("shell","monkey","-p",t,"1") or f"launched {t}"
        raise ValueError(f"Unsupported Android action: {k}")
