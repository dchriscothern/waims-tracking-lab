"""Physical load metrics per player-game (BUILD_SPEC.md section 4.2).

Primary time base is running clock (game clock not stopped). All distances in
feet, speeds in ft/s, accelerations in ft/s^2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FPS

# Speed zones, ft/s (lower bounds). Z4 + Z5 = high-speed distance.
ZONES = {"z1": 0.0, "z2": 7.0, "z3": 12.0, "z4": 16.0, "z5": 20.0}
HSD_MIN = ZONES["z4"]
SPRINT_MIN = ZONES["z5"]
SPRINT_MIN_S = 0.5
EFFORT_A = 9.8          # ft/s^2, about 3 m/s^2
EFFORT_MIN_S = 0.3


def run_starts(cond: np.ndarray, seg: np.ndarray, min_frames: int) -> np.ndarray:
    """Boolean array, True at the first frame of each run of `cond` that lasts
    at least `min_frames` frames without leaving its segment."""
    cond = np.asarray(cond, dtype=bool)
    seg = np.asarray(seg)
    n = len(cond)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    change = np.ones(n, dtype=bool)
    change[1:] = (cond[1:] != cond[:-1]) | (seg[1:] != seg[:-1])
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, n))
    keep = cond[starts] & (lengths >= min_frames)
    out[starts[keep]] = True
    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame flags: zone, sprint start, accel start, decel start."""
    df = df.copy()
    seg = df["seg"].to_numpy()
    v = df["v"].to_numpy()
    a = np.nan_to_num(df["a"].to_numpy(), nan=0.0)
    df["zone"] = pd.cut(df["v"], bins=list(ZONES.values()) + [np.inf],
                        labels=list(ZONES), right=False).astype(str)
    df["sprint_start"] = run_starts(v >= SPRINT_MIN, seg, round(SPRINT_MIN_S * FPS))
    df["accel_start"] = run_starts(a >= EFFORT_A, seg, round(EFFORT_MIN_S * FPS))
    df["decel_start"] = run_starts(a <= -EFFORT_A, seg, round(EFFORT_MIN_S * FPS))
    return df


def player_game(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game. Expects output of clean() + add_flags()."""
    run = df[df["running"]]
    d = run["v"] / FPS
    g = run.assign(d=d, d_hsd=np.where(run["v"] >= HSD_MIN, d, 0.0))
    for z in ZONES:
        g[f"d_{z}"] = np.where(g["zone"] == z, g["d"], 0.0)
    agg = g.groupby(["game_id", "side", "player_id"]).agg(
        frames=("frame", "size"),
        distance_ft=("d", "sum"),
        hsd_ft=("d_hsd", "sum"),
        **{f"{z}_ft": (f"d_{z}", "sum") for z in ZONES},
        sprints=("sprint_start", "sum"),
        accels=("accel_start", "sum"),
        decels=("decel_start", "sum"),
        peak_speed=("v", "max"),
        detected_share=("detected", "mean"),
        mean_pred_error=("pred_error", "mean"),
    )
    agg["minutes_running"] = agg.pop("frames") / FPS / 60
    all_min = df.groupby(["game_id", "side", "player_id"]).size() / FPS / 60
    agg["minutes_all"] = all_min.reindex(agg.index)
    m = agg["minutes_running"]
    agg["distance_per_min"] = agg["distance_ft"] / m
    agg["hsd_per_min"] = agg["hsd_ft"] / m
    agg["accels_per_min"] = agg["accels"] / m
    agg["decels_per_min"] = agg["decels"] / m
    agg["efforts_per_min"] = (agg["accels"] + agg["decels"]) / m
    det = run[run["detected"]].groupby(["game_id", "side", "player_id"])["v"].mean() * 60
    agg["distance_per_min_detected_only"] = det.reindex(agg.index)
    return agg.reset_index()


def by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Distance per running minute and efforts per minute, per player per period."""
    run = df[df["running"]].assign(d=lambda x: x["v"] / FPS,
                                   e=lambda x: x["accel_start"] | x["decel_start"])
    out = run.groupby(["game_id", "side", "player_id", "period"]).agg(
        frames=("frame", "size"), distance_ft=("d", "sum"), efforts=("e", "sum"))
    out["minutes"] = out.pop("frames") / FPS / 60
    out["distance_per_min"] = out["distance_ft"] / out["minutes"]
    out["efforts_per_min"] = out["efforts"] / out["minutes"]
    return out.reset_index()


def peak_windows(df: pd.DataFrame, seconds: int = 60, tol_s: float = 2.0) -> pd.DataFrame:
    """Rolling running-clock windows per player per period.

    A window is `seconds` of running-clock frames for one player within one
    period whose game-clock span is at most seconds + tol_s (so it never spans
    a stint on the bench). Returns every valid window's end row, with distance
    per minute, effort count, and start/end frames.
    """
    n = seconds * FPS
    run = df[df["running"]].sort_values(["player_id", "period", "frame"])
    rows = []
    for (gid, side, pid, per), g in run.groupby(["game_id", "side", "player_id", "period"], sort=False):
        if len(g) < n:
            continue
        d = np.cumsum(np.append(0.0, g["v"].to_numpy() / FPS))
        e = np.cumsum(np.append(0, (g["accel_start"] | g["decel_start"]).to_numpy().astype(int)))
        gc = g["game_clock"].to_numpy()
        fr = g["frame"].to_numpy()
        dist = d[n:] - d[:-n]
        eff = e[n:] - e[:-n]
        span = gc[: len(gc) - n + 1] - gc[n - 1:]
        ok = span <= seconds + tol_s
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        rows.append(pd.DataFrame({
            "game_id": gid, "side": side, "player_id": pid, "period": per,
            "start_frame": fr[idx], "end_frame": fr[idx + n - 1],
            "distance_per_min": dist[idx] * 60 / seconds,
            "efforts": eff[idx],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def top_windows(windows: pd.DataFrame, k: int = 3, seconds: int = 60) -> pd.DataFrame:
    """Top-k non-overlapping windows per player-game by distance per minute."""
    out = []
    for key, g in windows.sort_values("distance_per_min", ascending=False).groupby(
            ["game_id", "side", "player_id"], sort=False):
        chosen = []
        for r in g.itertuples(index=False):
            if all(r.end_frame < c.start_frame or r.start_frame > c.end_frame for c in chosen):
                chosen.append(r)
            if len(chosen) == k:
                break
        out.extend(c._asdict() | {"rank": i + 1} for i, c in enumerate(chosen))
    return pd.DataFrame(out)
