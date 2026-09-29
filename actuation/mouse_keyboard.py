"""Raksam Hands: reliable Windows mouse/keyboard primitives."""
from __future__ import annotations
import time
_pyautogui=None

def _pg():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.PAUSE=0.12
        _pyautogui=pyautogui
    return _pyautogui

def _active_window():
    """Return the foreground UIA window when running on Windows."""
    try:
        import win32gui
        from pywinauto import Application
        hwnd=win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        return Application(backend="uia").connect(handle=hwnd, timeout=2).window(handle=hwnd)
    except Exception:
        try:
            from pywinauto import Desktop
            return Desktop(backend="uia").window(active_only=True)
        except Exception:
            return None

def focus_active_text_control(click=True):
    """Focus the foreground window's largest editable/document control."""
    win=_active_window()
    if win is None:
        return False
    try: win.set_focus()
    except Exception: pass
    candidates=[]
    try:
        for ctrl in win.descendants():
            try:
                typ=str(ctrl.element_info.control_type or "").lower()
                if typ in {"edit","document","rich edit"} and ctrl.is_visible():
                    candidates.append(ctrl)
            except Exception:
                continue
    except Exception:
        candidates=[]
    if not candidates:
        return False
    def area(c):
        try:
            r=c.rectangle(); return max(0,r.width()*r.height())
        except Exception: return 0
    ctrl=max(candidates,key=area)
    try: ctrl.set_focus()
    except Exception: pass
    if click:
        try:
            ctrl.click_input()
        except Exception:
            try:
                r=ctrl.rectangle(); _pg().click(r.left+r.width()//2,r.top+r.height()//2)
            except Exception: pass
    time.sleep(.15)
    return True

def click(x,y,button="left"):
    p=_pg(); p.moveTo(int(x),int(y),duration=.15); p.click(button=button)

def double_click(x,y):
    p=_pg(); p.moveTo(int(x),int(y),duration=.15); p.doubleClick()

def move_mouse(x,y): _pg().moveTo(int(x),int(y),duration=.15)

def type_text(text,interval=.02):
    p=_pg()
    focused=focus_active_text_control(click=True)
    if not focused:
        # The foreground application may expose no UIA edit control. Keep the
        # keyboard path usable rather than silently pretending focus succeeded.
        try: p.click()
        except Exception: pass
    value=str(text)
    try:
        p.write(value,interval=interval)
    except Exception:
        try:
            import tkinter as tk
            r=tk.Tk(); r.withdraw(); r.clipboard_clear(); r.clipboard_append(value); r.update()
            p.hotkey("ctrl","v"); r.destroy()
        except Exception:
            raise

def press_key(key):
    p=_pg(); keys=[k.strip().lower() for k in str(key).split('+')]
    p.hotkey(*keys) if len(keys)>1 else p.press(keys[0])

def scroll(amount): _pg().scroll(int(amount))
def wait(seconds): time.sleep(max(0,float(seconds)))
