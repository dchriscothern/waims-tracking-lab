"""Closeouts: early vs late vs none, what they cost the defender and what they give up.

Population: catches that start in a three-point region. Each catch is one of
  closeout     SkillCorner recorded a closeout on the touch
  no_closeout  no closeout, assigned defender NO_CLOSEOUT_MIN_FT or more away at the catch
  guarded      no closeout, defender closer than that (no closeout needed; reference group)
Closeouts are split two ways: thirds of distance at the catch (near, mid, far),
and lead time, the seconds from closeout start to catch, at its median (early, late).
Catchers are also put in thirds of season 3-point attempts per game, to check
that no-closeout results are not just defenders sagging off non-shooters.

Defender cost uses the same window for every group, WINDOW_S around the catch.
The closeout endFrame in the data sits at catch + 0.8 s for almost every
closeout, before the defender has finished stopping, so it is not used.
Hard decels are counted relative to each defender's own ceiling (DECEL_CEIL_Q
quantile of detected running-clock braking, pooled across games).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pathlib import Path

from .io import FPS, RAW
from .metrics import EFFORT_A, EFFORT_MIN_S, run_starts
from .viz import INK, INK2, SURFACE, _finish, plt

PERIMETER = {"left corner three", "left wing three", "middle three", "right wing three", "right corner three"}
NO_CLOSEOUT_MIN_FT = 10.0
WINDOW_S = (-1.0, 2.0)
DECEL_PCTS = (0.70, 0.75, 0.80)
DECEL_CEIL_Q = 0.99
STOP_V = 3.0           # ft/s, "stopped"
STOP_MIN_PEAK = 7.0    # ft/s, only time a stop when the defender reached zone 2 speed


def _xy(v) -> tuple[float, float]:
    return (float(v[0]), float(v[1])) if isinstance(v, list) and len(v) >= 2 else (np.nan, np.nan)


def catches(ev: dict[str, pd.DataFrame], aliases: dict[int, int]) -> pd.DataFrame:
    """One row per perimeter catch: group, defender, distance and lead time, offensive outcome."""
    tou = ev["touches"].copy()
    tou["start_region"] = tou["regionsIn"].map(lambda r: r[0] if isinstance(r, list) and r else None)
    tou = tou[tou["start_region"].isin(PERIMETER)]
    loc = np.array([_xy(v) for v in tou["location"]])
    dl = np.array([_xy(v) for v in tou["closestDefLoc"]])
    out = pd.DataFrame({
        "game_id": tou["gameId"].to_numpy(), "touch_id": tou["id"].to_numpy(),
        "period": tou["period"].to_numpy(), "region": tou["start_region"].to_numpy(),
        "catch_frame": tou["startFrame"].to_numpy(),
        "player_id": tou["playerId"].to_numpy(), "defender_id": tou["defenderId"].to_numpy(),
        "dist_catch": np.hypot(loc[:, 0] - dl[:, 0], loc[:, 1] - dl[:, 1]),
        "outcomes": tou["outcomes"].to_numpy(),
    })

    co = ev["closeouts"].drop_duplicates("touchId").set_index("touchId")
    has = out["touch_id"].isin(co.index)
    c = co.reindex(out.loc[has, "touch_id"])
    out.loc[has, "defender_id"] = c["ballhandlerDefId"].to_numpy()
    out.loc[has, "dist_catch"] = c["touchDistance"].to_numpy()
    out["lead_s"] = np.nan
    out.loc[has, "lead_s"] = ((c["touchFrame"] - c["startFrame"]) / FPS).to_numpy()
    out["end_distance"] = np.nan
    out.loc[has, "end_distance"] = c["endDistance"].to_numpy()
    out["group"] = np.where(has, "closeout",
                            np.where(out["dist_catch"] >= NO_CLOSEOUT_MIN_FT, "no_closeout", "guarded"))

    # Offensive outcome of the catch
    sh = ev["shots"].dropna(subset=["touchId"]).drop_duplicates("touchId").set_index("touchId")
    s = sh.reindex(out["touch_id"])
    out["shot"] = s["id"].notna().to_numpy()
    out["three"] = s["three"].fillna(False).astype(bool).to_numpy()
    out["made"] = s["outcome"].fillna(False).astype(bool).to_numpy()
    out["points"] = np.where(out["made"], np.where(out["three"], 3, 2), 0)
    out["shot_quality"] = s["shotQuality"].to_numpy(dtype=float)
    out["def_dist_release"] = s["closestDefDist"].to_numpy(dtype=float)
    dr = ev["drives"].dropna(subset=["touchId"]).groupby("touchId")["blowby"].max()
    out["drive"] = out["touch_id"].isin(dr.index)
    out["blowby"] = out["touch_id"].map(dr).fillna(False).astype(bool)
    out["action"] = np.select(
        [out["shot"], out["drive"], out["outcomes"].map(lambda o: isinstance(o, list) and "PASS" in o)],
        ["shot", "drive", "pass"], "other")
    out = out.drop(columns="outcomes").dropna(subset=["defender_id", "dist_catch", "catch_frame"])
    for col in ("player_id", "defender_id"):
        out[col] = out[col].astype(int).map(lambda p: aliases.get(p, p))
    return out.reset_index(drop=True)


def traces(cat: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """Defender v, a and detected flag in the window around each catch (one game)."""
    lo_f, hi_f = round(WINDOW_S[0] * FPS), round(WINDOW_S[1] * FPS)
    by_player = df.groupby("player_id").indices
    frames, v, a, det = df["frame"].to_numpy(), df["v"].to_numpy(), df["a"].to_numpy(), df["detected"].to_numpy()
    out = []
    for r in cat.itertuples(index=False):
        idx = by_player.get(r.defender_id)
        if idx is None:
            out.append(None)
            continue
        f = frames[idx]
        lo = np.searchsorted(f, r.catch_frame + lo_f)
        hi = np.searchsorted(f, r.catch_frame + hi_f, side="right")
        sl = idx[lo:hi]
        out.append({"frame": frames[sl], "v": v[sl], "a": a[sl], "detected": det[sl]} if len(sl) >= 10 else None)
    return out


def decel_ceiling(df: pd.DataFrame) -> pd.Series:
    """Detected running-clock braking values (positive ft/s^2) per player, for pooling across games."""
    m = df["running"] & df["detected"] & (df["a"] < 0)
    return (-df.loc[m, "a"]).groupby(df.loc[m, "player_id"]).apply(np.asarray)


def cost(tr: dict | None, ceiling: float) -> dict:
    """Defender load in the catch window."""
    if tr is None:
        return {}
    v, a = tr["v"], np.nan_to_num(tr["a"], nan=0.0)
    seg = np.cumsum(np.diff(tr["frame"], prepend=tr["frame"][0] - 1) != 1)
    brake = -a
    k = int(np.argmax(brake))
    out = {
        "window_frames": len(v),
        "detected_share": float(np.mean(tr["detected"])),
        "peak_speed": float(v.max()),
        "distance_ft": float(v.sum() / FPS),
        "peak_decel": float(brake.max()),
        "entry_speed": float(v[: k + 1].max()),
        "speed_shed": float(brake[brake >= EFFORT_A].sum() / FPS),
        "hard_decels_abs": int(run_starts(brake >= EFFORT_A, seg, round(EFFORT_MIN_S * FPS)).sum()),
        "stop_time_s": np.nan, "stop_dist_ft": np.nan,
    }
    # Stopping: from peak speed to the first frame under STOP_V, if the defender got moving and stopped in the window
    j = int(np.argmax(v))
    if v[j] >= STOP_MIN_PEAK:
        below = np.flatnonzero(v[j:] < STOP_V)
        if len(below) and seg[j] == seg[j + below[0]]:
            out["stop_time_s"] = below[0] / FPS
            out["stop_dist_ft"] = float(v[j:j + below[0]].sum() / FPS)
    for p in DECEL_PCTS:
        out[f"hard_decels_{round(p * 100)}"] = (
            int(run_starts(brake >= p * ceiling, seg, round(EFFORT_MIN_S * FPS)).sum())
            if np.isfinite(ceiling) else np.nan)
    return out


def add_splits(cat: pd.DataFrame) -> pd.DataFrame:
    """Thirds of distance at the catch and of lead time, closeouts only."""
    cat = cat.copy()
    co = cat["group"] == "closeout"
    cat["dist_split"] = cat["group"]
    cat.loc[co, "dist_split"] = pd.qcut(cat.loc[co, "dist_catch"], 3, labels=["near", "mid", "far"]).astype(str)
    # Lead time comes in a few discrete values (many ties at 0.40 s), so thirds are lopsided: split at the median
    cat["lead_split"] = cat["group"]
    med = cat.loc[co, "lead_s"].median()
    cat.loc[co, "lead_split"] = np.where(cat.loc[co, "lead_s"] > med, "early", "late")
    return cat


def shooter_volume(aliases: dict[int, int]) -> pd.Series:
    """Season 3-point attempts per game by player, from the ACB season aggregates (about 293 games,
    so nearly independent of the 10 tracked games)."""
    a = pd.read_csv(RAW / "aggregates" / "acb_shotsaggregates_20252026.csv")
    a["player_id"] = a["player_id"].astype(int).map(lambda p: aliases.get(p, p))
    g = a.groupby("player_id")[["three_attempts", "games_played"]].sum()
    return (g["three_attempts"] / g["games_played"]).rename("season_3pa_per_game")


def add_shooter_tier(cat: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Thirds of the catcher's season 3PA per game, cut over all perimeter catches."""
    cat = cat.copy()
    cat["season_3pa_per_game"] = cat["player_id"].map(vol)
    cat["shooter_tier"] = pd.qcut(cat["season_3pa_per_game"], 3, labels=["low", "mid", "high"]).astype(str)
    return cat


def shooter_check(cat: pd.DataFrame) -> pd.DataFrame:
    """Shot rate and points per 100 catches by shooter tier and group (distance split)."""
    d = cat[cat["shooter_tier"].isin(["low", "mid", "high"])]
    g = d.groupby(["shooter_tier", "dist_split"]).agg(
        n=("touch_id", "size"), shot_rate=("shot", "mean"), pts_per_100=("points", "mean"),
        def_dist_release=("def_dist_release", "mean"))
    g["pts_per_100"] *= 100
    g["share_of_tier"] = g["n"] / g.groupby(level=0)["n"].transform("sum")
    return g.reset_index()


SUMMARY_METRICS = {
    # name: (column, aggregation over catches)
    "hard_decels_75": ("hard_decels_75", "mean"),
    "peak_decel": ("peak_decel", "mean"),
    "speed_shed": ("speed_shed", "mean"),
    "peak_speed": ("peak_speed", "mean"),
    "entry_speed": ("entry_speed", "mean"),
    "stop_time_s": ("stop_time_s", "mean"),
    "stop_dist_ft": ("stop_dist_ft", "mean"),
    "shot_rate": ("shot", "mean"),
    "pts_per_100": ("points", "mean"),
    "shot_quality": ("shot_quality", "mean"),
    "def_dist_release": ("def_dist_release", "mean"),
    "blowby_rate": ("blowby", "mean"),
}


def summary(cat: pd.DataFrame, split: str, n_boot: int = 1000, seed: int = 11) -> pd.DataFrame:
    """Group means with 95% intervals from a bootstrap that resamples games."""
    rng = np.random.default_rng(seed)
    games = cat["game_id"].unique()
    by_game = {g: d for g, d in cat.groupby("game_id")}

    def means(d: pd.DataFrame) -> pd.DataFrame:
        m = d.groupby(split).agg(**{k: (c, f) for k, (c, f) in SUMMARY_METRICS.items()})
        m["pts_per_100"] *= 100
        return m

    point = means(cat)
    point.insert(0, "n", cat.groupby(split).size())
    boots = [means(pd.concat([by_game[g] for g in rng.choice(games, len(games))])) for _ in range(n_boot)]
    stack = pd.concat(boots, keys=range(n_boot))
    lo = stack.groupby(level=1).quantile(0.025).add_suffix("_lo")
    hi = stack.groupby(level=1).quantile(0.975).add_suffix("_hi")
    out = point.join(lo).join(hi)
    out = out[["n"] + [c for k in SUMMARY_METRICS for c in (k, f"{k}_lo", f"{k}_hi")]]
    order = [o for o in ["guarded", "near", "mid", "far", "late", "early", "no_closeout"] if o in out.index]
    out = out.reindex(order + sorted(set(out.index) - set(order)))
    return out.rename_axis("group").reset_index().assign(split=split)


DRIFT_METRICS = ["peak_decel", "entry_speed", "peak_speed", "stop_time_s", "stop_dist_ft", "hard_decels_75"]


def quarter_drift(cat: pd.DataFrame, n_boot: int = 1000, seed: int = 11) -> pd.DataFrame:
    """Q4 minus Q1 within the same defender and game, closeouts only (a fatigue check).

    Each defender-game with closeouts in both quarters gives one paired difference
    of its quarter means. Interval from a bootstrap that resamples games.
    """
    co = cat[(cat["group"] == "closeout") & cat["period"].isin([1, 4])]
    m = co.groupby(["game_id", "defender_id", "period"])[DRIFT_METRICS].mean().unstack("period")
    rows = []
    rng = np.random.default_rng(seed)
    for k in DRIFT_METRICS:
        d = (m[(k, 4)] - m[(k, 1)]).dropna()
        if d.empty:
            continue
        by_game = d.groupby(level="game_id")
        games = list(by_game.groups)
        boots = [pd.concat([by_game.get_group(g) for g in rng.choice(games, len(games))]).mean()
                 for _ in range(n_boot)]
        rows.append({"metric": k, "pairs": len(d), "q1_mean": m[(k, 1)][d.index].mean(),
                     "q4_minus_q1": d.mean(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975)})
    return pd.DataFrame(rows)


def correlations(cat: pd.DataFrame) -> pd.DataFrame:
    """Spearman's rho among closeouts: distance and lead time against cost and outcome."""
    co = cat[cat["group"] == "closeout"]
    rows = []
    for x in ("dist_catch", "lead_s"):
        for y in ("hard_decels_75", "peak_decel", "speed_shed", "end_distance", "def_dist_release",
                  "shot_quality", "points"):
            d = co[[x, y]].dropna()
            rho, p = spearmanr(d[x], d[y])
            rows.append({"x": x, "y": y, "n": len(d), "rho": rho, "p": p})
    return pd.DataFrame(rows)


def figure(summ: pd.DataFrame, out: Path) -> None:
    s = summ[summ["split"] == "dist_split"].set_index("group")
    names = {"guarded": "Already guarded (under 10 ft)", "near": "Closeout, near third",
             "mid": "Closeout, middle third", "far": "Closeout, far third", "no_closeout": "No closeout (10+ ft)"}
    s = s.reindex([g for g in names if g in s.index])
    y = np.arange(len(s))[::-1]
    colors = ["#2a78d6" if g in ("near", "mid", "far") else INK2 for g in s.index]
    panels = [("hard_decels_75", "Defender hard decels per catch\n(75% of own max, 1 s before to 2 s after)"),
              ("pts_per_100", "Field goal points per 100 catches"),
              ("shot_quality", "Shot quality when a shot is taken\n(SkillCorner, 0 to 100)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    for ax, (col, label) in zip(axes, panels):
        for yi, (g, r), c in zip(y, s.iterrows(), colors):
            ax.plot([r[f"{col}_lo"], r[f"{col}_hi"]], [yi, yi], color=c, lw=1.5, alpha=0.6)
            ax.plot(r[col], yi, "o", color=c, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r[col]:.2f}" if r[col] < 10 else f"{r[col]:.0f}", (r[col], yi), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(s) - 0.3)
    axes[0].set_yticks(y, [f"{names[g]}  n={int(n)}" for g, n in zip(s.index, s["n"])])
    out.mkdir(parents=True, exist_ok=True)
    _finish(fig, out / "closeouts.png", "Closeouts: what they cost the defender and what they give up",
            "Catches in three-point regions. Closeouts split into thirds by defender distance at the catch. "
            "Lines = 95% interval, bootstrap by game.")


def early_figure(summ: pd.DataFrame, out: Path) -> None:
    """Early vs late closeouts: the defender pays more, the shot does not change."""
    s = summ[summ["split"] == "lead_split"].set_index("group").reindex(["late", "early"])
    names = {"late": "Late start\n(0.40 s or less before the catch)", "early": "Early start\n(more than 0.40 s before)"}
    panels = [("peak_decel", "Defender peak braking\n(ft/s²)", "cost"),
              ("hard_decels_75", "Defender hard decels\nper closeout", "cost"),
              ("shot_quality", "Shot quality when a shot\nis taken (0 to 100)", "result"),
              ("pts_per_100", "Field goal points\nper 100 catches", "result")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    y = np.array([1, 0])
    for ax, (col, label, kind) in zip(axes, panels):
        c = "#e34948" if kind == "cost" else INK2
        for yi, (g, r) in zip(y, s.iterrows()):
            ax.plot([r[f"{col}_lo"], r[f"{col}_hi"]], [yi, yi], color=c, lw=1.5, alpha=0.6)
            ax.plot(r[col], yi, "o", color=c, ms=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r[col]:.2f}" if r[col] < 10 else f"{r[col]:.1f}" if r[col] < 30 else f"{r[col]:.0f}",
                        (r[col], yi), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9,
                        color=INK)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, 1.6)
    axes[0].set_yticks(y, [f"{names[g]}\nn={int(n)}" for g, n in zip(s.index, s["n"])])
    out.mkdir(parents=True, exist_ok=True)
    _finish(fig, out / "early_closeouts.png", "Jumping the pass costs the defender's legs, not the shooter's shot",
            "Perimeter closeouts in 10 Liga ACB games, split at the median start time. Red = cost to the defender, "
            "gray = result. Lines = 95% interval, bootstrap by game.")
