"""
perception/screen_state.py

READY-MADE PIECE (not your custom brain).
Turns the raw Windows screen into a short, structured TEXT description
that your brain model can read. This is what lets your brain avoid
needing to understand raw pixels.

Two sources of information, combined:
  1. UI Automation (via pywinauto) -> real clickable elements with
     names, types, and exact coordinates. Much more reliable than
     guessing from a screenshot.
  2. OCR (via pytesseract) -> fallback text extraction for apps that
     don't expose a proper UI Automation tree (canvas apps, some games).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

try:
    import mss
except Exception:
    mss = None
try:
    import pytesseract
except Exception:
    pytesseract = None
from PIL import Image

try:
    from pywinauto import Desktop
except ImportError:
    Desktop = None  # pywinauto is Windows-only; keep import optional for non-Windows dev/testing


@dataclass
class UIElement:
    control_type: str      # e.g. "Button", "Edit", "MenuItem"
    name: str               # visible label / accessible name
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def to_text(self) -> str:
        cx, cy = self.center
        return f'[{self.control_type} "{self.name}" @ ({cx},{cy})]'


@dataclass
class ScreenState:
    active_window_title: str
    elements: list[UIElement] = field(default_factory=list)
    ocr_text: str = ""
    screenshot_path: str | None = None

    def to_prompt_text(self, max_elements: int = 40) -> str:
        """Serialize into the short text format the brain model reads."""
        lines = [f"WINDOW: {self.active_window_title}"]
        if self.elements:
            lines.append("ELEMENTS:")
            for el in self.elements[:max_elements]:
                lines.append("  " + el.to_text())
        if self.ocr_text.strip():
            snippet = self.ocr_text.strip().replace("\n", " ")[:300]
            lines.append(f"OCR_TEXT: {snippet}")
        return "\n".join(lines)


def capture_screenshot(save_path: str | None = None) -> Image.Image:
    """Grab a screenshot of the primary monitor using mss (fast, no GDI leaks)."""
    if mss is None:
        raise RuntimeError("mss is not installed; install project requirements for screen capture")
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # index 0 = all monitors combined; 1 = primary
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        if save_path:
            img.save(save_path)
        return img


def get_ui_elements(max_depth: int = 3) -> tuple[str, list[UIElement]]:
    """
    Walk the UI Automation tree of the foreground window.
    Returns (window_title, elements). Falls back to empty list if
    pywinauto/UIA is unavailable (e.g. running on non-Windows for dev).
    """
    if Desktop is None:
        return "UNKNOWN (pywinauto not available)", []

    try:
        desktop = Desktop(backend="uia")
        win = desktop.window(active_only=True)
        title = win.window_text() or "Untitled"

        elements: list[UIElement] = []

        def walk(ctrl, depth):
            if depth > max_depth:
                return
            for child in ctrl.children():
                try:
                    rect = child.rectangle()
                    name = child.window_text() or child.element_info.name or ""
                    ctrl_type = child.element_info.control_type
                    if name.strip():
                        elements.append(
                            UIElement(
                                control_type=ctrl_type,
                                name=name.strip(),
                                left=rect.left,
                                top=rect.top,
                                right=rect.right,
                                bottom=rect.bottom,
                            )
                        )
                    walk(child, depth + 1)
                except Exception:
                    # Some elements throw on access (invisible/stale) - skip them
                    continue

        walk(win, 0)
        return title, elements
    except Exception as e:
        return f"UNKNOWN (error: {e})", []


def get_ocr_text(img: Image.Image) -> str:
    """OCR fallback - only useful when UIA element list came back thin."""
    try:
        if pytesseract is None:
            return ""
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def perceive(save_screenshot_to: str | None = None, run_ocr_if_elements_below: int = 3) -> ScreenState:
    """
    Main entry point: capture the full current screen state.

    run_ocr_if_elements_below: only run (slower) OCR when the UIA tree
    returned very few elements - keeps perception fast in the common case.
    """
    title, elements = get_ui_elements()
    img = capture_screenshot(save_screenshot_to)

    ocr_text = ""
    if len(elements) < run_ocr_if_elements_below:
        ocr_text = get_ocr_text(img)

    return ScreenState(
        active_window_title=title,
        elements=elements,
        ocr_text=ocr_text,
        screenshot_path=save_screenshot_to,
    )


if __name__ == "__main__":
    state = perceive(save_screenshot_to="debug_screenshot.png")
    print(state.to_prompt_text())
