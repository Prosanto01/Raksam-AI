"""Windows-native speech input and output without third-party audio packages."""
from __future__ import annotations

import base64
import os
import subprocess


def _powershell(script: str, timeout: float) -> str:
    if os.name != "nt":
        raise RuntimeError("Windows voice mode is available only on Windows.")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=timeout, check=False,
        creationflags=flags,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "Windows speech service failed.")
    return proc.stdout.strip()


def speak(text: str, rate: int = 0) -> None:
    """Speak plain text through the current Windows default voice."""
    encoded = base64.b64encode(str(text).encode("utf-16-le")).decode("ascii")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        f"$t=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded}')); "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate={max(-10, min(10, int(rate)))}; $s.Speak($t)"
    )
    _powershell(script, timeout=max(15, len(str(text)) / 8))


def listen_once(timeout_seconds: int = 12) -> str:
    """Listen once using Windows dictation and return recognized text."""
    seconds = max(1, min(60, int(timeout_seconds)))
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$r=New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
        "$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
        "$r.SetInputToDefaultAudioDevice(); "
        f"$x=$r.Recognize([TimeSpan]::FromSeconds({seconds})); "
        "if($x){[Console]::Out.Write($x.Text)}"
    )
    return _powershell(script, timeout=seconds + 10)
