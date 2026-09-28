"""Clean tracking: segment, cap, smooth speed, derive acceleration.

Method (see BUILD_SPEC.md section 4.1):
  1. Split each player's frames into continuous segments at frame gaps.
  2. Cap the provided speed at SPEED_CAP ft/s (rare artifact spikes).
  3. Savitzky Golay smooth inside each segment.
  4. Acceleration = first difference of smoothed speed times FPS, inside segments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

from .io import FPS

SPEED_CAP = 26.0      # ft/s, about 7.9 m/s
SG_WINDOW = 13        # frames, 0.52 s
SG_POLY = 2


def _smooth(v: np.ndarray, method: str, window: int) -> np.ndarray:
    if method == "savgol":
        if len(v) < window:
            return v
        return savgol_filter(v, window, SG_POLY)
    if method == "butter":
        b, a = butter(4, 1.0 / (FPS / 2))           # 4th order, 1 Hz low-pass
        if len(v) <= 3 * max(len(a), len(b)):
            return v
        return filtfilt(b, a, v)
    if method == "none":
        return v
    raise ValueError(method)


def clean(df: pd.DataFrame, aliases: dict[int, int] | None = None,
          method: str = "savgol", window: int = SG_WINDOW) -> pd.DataFrame:
    """Add seg, running, v (smoothed ft/s), a (ft/s^2) and capped flag."""
    df = df.sort_values(["player_id", "frame"], ignore_index=True).copy()
    if aliases:
        df["player_id"] = df["player_id"].map(lambda p: aliases.get(p, p)).astype("int32")
    new_seg = (df["player_id"].diff() != 0) | (df["frame"].diff() != 1)
    df["seg"] = new_seg.cumsum().astype("int32")
    df["running"] = ~df["clock_stopped"]
    df["capped"] = df["speed"] > SPEED_CAP
    raw = df["speed"].clip(upper=SPEED_CAP).to_numpy(dtype="float64")

    v = np.empty_like(raw)
    starts = np.flatnonzero(new_seg.to_numpy())
    bounds = np.append(starts, len(raw))
    for s, e in zip(bounds[:-1], bounds[1:]):
        v[s:e] = _smooth(raw[s:e], method, window)
    v = np.clip(v, 0, SPEED_CAP)
    df["v"] = v.astype("float32")

    a = np.diff(v, prepend=np.nan) * FPS
    a[starts] = np.nan
    df["a"] = a.astype("float32")
    return df
