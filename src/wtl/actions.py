"""Physical cost of tactical actions (BUILD_SPEC.md section 4.4).

For each action, each involved player's load is measured inside the action
window and compared with that player's own running-clock average speed in the
same game ("excess" values). That controls for fast players always being fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FPS

POINT_WINDOW_S = 2.0     # +/- seconds around single-frame events

# action table -> (window type, [(actor label, player col, side, coverage col)])
ACTIONS = {
    "drives": ("span", [("ballhandler", "ballhandlerId", "off", None),
                        ("ballhandler_def", "ballhandlerDefId", "def", None)]),
    "isolations": ("span", [("ballhandler", "ballhandlerId", "off", None),
                            ("ballhandler_def", "ballhandlerDefId", "def", None)]),
    "posts": ("span", [("ballhandler", "ballhandlerId", "off", None),
                       ("ballhandler_def", "ballhandlerDefId", "def", None)]),
    "closeouts": ("span", [("ballhandler", "ballhandlerId", "off", None),
                           ("ballhandler_def", "ballhandlerDefId", "def", None)]),
    "picks": ("point", [("ballhandler", "ballhandlerId", "off", "bhrDefType"),
                        ("screener", "screenerId", "off", "scrDefType"),
                        ("ballhandler_def", "ballhandlerDefId", "def", "bhrDefType"),
                        ("screener_def", "screenerDefId", "def", "scrDefType")]),
    "handoffs": ("point", [("setter", "setterId", "off", "setterDefType"),
                           ("receiver", "receiverId", "off", "receiverDefType"),
                           ("setter_def", "setterDefId", "def", "setterDefType"),
                           ("receiver_def", "receiverDefId", "def", "receiverDefType")]),
    "off_ball_screens": ("point", [("cutter", "cutterId", "off", "cutterDefType"),
                                   ("screener", "screenerId", "off", "screenerDefType"),
                                   ("cutter_def", "cutterDefId", "def", "cutterDefType"),
                                   ("screener_def", "screenerDefId", "def", "screenerDefType")]),
}
EXTRA_COLS = {"drives": ["blowby", "category", "endType"], "closeouts": ["bhrAction"],
              "picks": ["direct"], "handoffs": ["direct"], "off_ball_screens": []}


def action_rows(events: dict[str, pd.DataFrame], aliases: dict[int, int]) -> pd.DataFrame:
    """One row per (action, involved player) with its frame window."""
    half = int(POINT_WINDOW_S * FPS)
    rows = []
    for name, (kind, actors) in ACTIONS.items():
        ev = events.get(name)
        if ev is None or ev.empty:
            continue
        if kind == "span":
            start, end = ev["startFrame"], ev["endFrame"]
        else:
            start, end = ev["frame"] - half, ev["frame"] + half
        for label, pcol, side, cov in actors:
            if pcol not in ev:
                continue
            df = pd.DataFrame({
                "game_id": ev["gameId"], "action": name, "action_id": ev["id"],
                "actor": label, "actor_side": side, "player_id": ev[pcol],
                "start_frame": start, "end_frame": end,
                "coverage": ev[cov] if cov and cov in ev else None,
            })
            for c in EXTRA_COLS.get(name, []):
                if c in ev:
                    df[c] = ev[c]
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["player_id", "start_frame", "end_frame"])
    out = out[out["end_frame"] > out["start_frame"]]
    out["player_id"] = out["player_id"].astype(int).map(lambda p: aliases.get(p, p))
    return out.reset_index(drop=True)


def measure(acts: pd.DataFrame, df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """Load inside each action window for its player.

    df: cleaned + flagged tracking for ONE game.
    baseline: running-clock mean speed (ft/s) by player_id for that game.
    """
    acts = acts.copy()
    res = {k: np.full(len(acts), np.nan) for k in
           ("frames", "mean_speed", "peak_speed", "distance_ft", "max_accel", "max_decel", "efforts")}
    eff = (df["accel_start"] | df["decel_start"]).to_numpy()
    by_player = {pid: idx for pid, idx in df.groupby("player_id").indices.items()}
    frames = df["frame"].to_numpy()
    v = df["v"].to_numpy()
    a = df["a"].to_numpy()
    for i, r in enumerate(acts.itertuples(index=False)):
        idx = by_player.get(r.player_id)
        if idx is None:
            continue
        f = frames[idx]
        lo, hi = np.searchsorted(f, r.start_frame), np.searchsorted(f, r.end_frame, side="right")
        if hi - lo < 3:
            continue
        sl = idx[lo:hi]
        res["frames"][i] = hi - lo
        res["mean_speed"][i] = v[sl].mean()
        res["peak_speed"][i] = v[sl].max()
        res["distance_ft"][i] = v[sl].sum() / FPS
        aa = a[sl]
        if np.isfinite(aa).any():
            res["max_accel"][i] = np.nanmax(aa)
            res["max_decel"][i] = np.nanmin(aa)
        res["efforts"][i] = eff[sl].sum()
    for k, arr in res.items():
        acts[k] = arr
    acts["duration_s"] = acts["frames"] / FPS
    acts["baseline_speed"] = acts["player_id"].map(baseline)
    acts["excess_speed"] = acts["mean_speed"] - acts["baseline_speed"]
    acts["excess_distance_ft"] = acts["distance_ft"] - acts["baseline_speed"] * acts["duration_s"]
    acts["efforts_per_min"] = acts["efforts"] / acts["duration_s"] * 60
    return acts.dropna(subset=["mean_speed"])


def cost_table(measured: pd.DataFrame, roles: pd.Series) -> pd.DataFrame:
    """Aggregate by action, actor and player role."""
    m = measured.assign(player_role=measured["player_id"].map(roles)).dropna(subset=["player_role"])
    out = m.groupby(["action", "actor", "player_role"]).agg(
        n=("action_id", "size"),
        mean_duration_s=("duration_s", "mean"),
        mean_speed=("mean_speed", "mean"),
        mean_excess_speed=("excess_speed", "mean"),
        mean_excess_distance_ft=("excess_distance_ft", "mean"),
        efforts_per_min=("efforts_per_min", "mean"),
        mean_peak_speed=("peak_speed", "mean"),
    ).reset_index()
    return out[out["n"] >= 10]
