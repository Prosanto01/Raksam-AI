"""Compatibility wrapper for the Raksam Hands layer."""
from __future__ import annotations
from actuation import mouse_keyboard as low
class MouseKeyboardHands:
    move=staticmethod(low.move_mouse); click=staticmethod(low.click); double_click=staticmethod(low.double_click)
    type=staticmethod(low.type_text); press=staticmethod(low.press_key); scroll=staticmethod(low.scroll); wait=staticmethod(low.wait)
