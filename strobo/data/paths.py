"""Locate the public datasets under a root (Kaggle: /kaggle/input).  Each finder
returns the first directory that contains the expected files, or None."""
from __future__ import annotations

import glob
import os


def _first_dir(root: str, pattern: str, up: int = 0):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        return None
    p = hits[0]
    for _ in range(up):
        p = os.path.dirname(p)
    return p


def find_dalia(root: str):
    """PPG-DaLiA: pickles S1/S1.pkl ... S15/S15.pkl (contain 'wrist' + 'chest' + 'rpeaks')."""
    for h in sorted(glob.glob(os.path.join(root, "**", "S1", "S1.pkl"), recursive=True)):
        d = os.path.dirname(os.path.dirname(h))
        if "dalia" in d.lower() or "ppg" in d.lower() or os.path.exists(os.path.join(d, "S15", "S15.pkl")):
            return d
    return None


def find_wesad(root: str):
    """WESAD: pickles S2/S2.pkl ... S17/S17.pkl (there is no S1 / S12)."""
    for h in sorted(glob.glob(os.path.join(root, "**", "S2", "S2.pkl"), recursive=True)):
        d = os.path.dirname(os.path.dirname(h))
        if "wesad" in d.lower() or os.path.exists(os.path.join(d, "S17", "S17.pkl")):
            return d
    return None


def find_har(root: str):
    return _first_dir(root, "Inertial Signals", up=2)


def find_mitbih(root: str):
    for h in sorted(glob.glob(os.path.join(root, "**", "100.hea"), recursive=True)):
        if os.path.exists(h[:-4] + ".atr"):
            return os.path.dirname(h)
    return None


def find_all(root: str) -> dict:
    return {"ppg_dalia": find_dalia(root), "wesad": find_wesad(root), "uci_har": find_har(root),
            "mitbih": find_mitbih(root)}
