"""Event-derived role buckets (BUILD_SPEC.md section 4.3).

Positions are not published, so roles come from what players do on court,
pooled across all games. Two methods are computed and compared:
  * k-means (k = 3) on standardized per-36 features, clusters named by centroid
  * a transparent scoring rule (guard score vs big score)
The rule is the default label because it is explainable; k-means agreement is
reported as a check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

HOOP_X = -40.76      # ft, offensive hoop in event coordinates (FIBA court 28 m)
MIN_MINUTES = 10.0

FEATURES = ["touches", "drives", "picks_bh", "picks_scr", "obs_scr", "posts",
            "rebounds", "rim_share", "three_share", "touch_dist_hoop"]
RATE_FEATURES = FEATURES[:7]


def _count(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(dtype=float, name=name)
    return df[col].dropna().astype(int).value_counts().rename(name)


def features(events_by_game: dict[int, dict[str, pd.DataFrame]], minutes: pd.Series,
             aliases: dict[int, int]) -> pd.DataFrame:
    """Per-player features. `minutes` is running-clock minutes indexed by player_id."""
    cat = {k: pd.concat([ev[k] for ev in events_by_game.values()], ignore_index=True)
           for k in ("touches", "drives", "picks", "off_ball_screens", "posts", "rebounds", "shots")}
    player_cols = ("playerId", "ballhandlerId", "screenerId", "rebounderId", "shooterId")
    for k, df in cat.items():
        for c in player_cols:
            if c in df:
                df[c] = df[c].map(lambda p: aliases.get(int(p), int(p)) if pd.notna(p) else p)
    parts = [
        _count(cat["touches"], "playerId", "touches"),
        _count(cat["drives"], "ballhandlerId", "drives"),
        _count(cat["picks"], "ballhandlerId", "picks_bh"),
        _count(cat["picks"], "screenerId", "picks_scr"),
        _count(cat["off_ball_screens"], "screenerId", "obs_scr"),
        _count(cat["posts"], "ballhandlerId", "posts"),
        _count(cat["rebounds"], "rebounderId", "rebounds"),
    ]
    f = pd.concat(parts, axis=1).fillna(0)
    shots = cat["shots"].dropna(subset=["shooterId"])
    shots = shots.assign(shooterId=shots["shooterId"].astype(int))
    grp = shots.groupby("shooterId")
    f["rim_share"] = grp.apply(lambda s: (s["region"] == "ra").mean(), include_groups=False)
    f["three_share"] = grp["three"].mean()
    t = cat["touches"].dropna(subset=["location"])
    t = t.assign(playerId=t["playerId"].astype(int),
                 dist=[np.hypot(l[0] - HOOP_X, l[1]) for l in t["location"]])
    f["touch_dist_hoop"] = t.groupby("playerId")["dist"].median()
    f = f.join(minutes.rename("minutes"), how="right").fillna(0)
    f = f[f["minutes"] >= MIN_MINUTES].copy()
    for c in RATE_FEATURES:
        f[c] = f[c] / f["minutes"] * 36
    f.index.name = "player_id"
    return f


def assign(f: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add role_rule, role_kmeans and role (= role_rule)."""
    z = (f[FEATURES] - f[FEATURES].mean()) / f[FEATURES].std(ddof=0)
    big = z[["picks_scr", "obs_scr", "posts", "rebounds", "rim_share"]].mean(axis=1) \
        - z[["three_share", "touch_dist_hoop"]].mean(axis=1)
    guard = z[["touches", "drives", "picks_bh"]].mean(axis=1)
    out = f.copy()
    out["big_score"], out["guard_score"] = big, guard
    rule = np.where((big > 0.25) & (big > guard), "big",
                    np.where((guard > 0.25) & (guard > big), "guard", "wing"))
    out["role_rule"] = rule

    np.random.seed(seed)
    cent, lab = kmeans2(z.to_numpy(), 3, minit="++", seed=seed)
    cent = pd.DataFrame(cent, columns=FEATURES)
    g = int(cent["picks_bh"].idxmax())
    b = int((cent["picks_scr"] + cent["rebounds"] + cent["rim_share"]).drop(g).idxmax())
    names = {g: "guard", b: "big"}
    out["role_kmeans"] = [names.get(int(l), "wing") for l in lab]
    out["role"] = out["role_rule"]
    return out


def summary(roles: pd.DataFrame) -> pd.DataFrame:
    """Mean features per role, for the README sanity check."""
    return roles.groupby("role")[FEATURES + ["minutes"]].mean().round(2).assign(
        n=roles.groupby("role").size())
