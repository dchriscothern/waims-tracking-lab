"""Load before a shot vs shot execution (pick 2).

For each field goal attempt, the shooter's load in the last WINDOWS_S seconds of
running clock before the release (same period, never spanning a bench stint).
Load: distance per minute, and hard decels per minute at DECEL_PCT of the
player's own braking ceiling (as in closeouts.py).

Outcomes: shot quality, shot selection (three, catch and shoot, contested,
release time) and make vs expected (made minus a logistic fit of make on
shot quality and 2 vs 3, fitted on all shots).

Within-player tests use the player-game as the subject: repeated-measures
correlation (rmcorr, Bakdash and Marusich 2017) and outcome means across
within-player load bins, with intervals from a bootstrap that resamples games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .io import FPS
from .metrics import EFFORT_MIN_S, run_starts

WINDOWS_S = (60, 120)
DECEL_PCT = 0.75
SPAN_TOL_S = 2.0
LOAD_COLS = [f"{m}_{w}s" for w in WINDOWS_S for m in ("dist_per_min", "decels_per_min")]
OUTCOMES = ["shot_quality", "make_vs_expected", "three", "catch_and_shoot", "contested", "release_time",
            "shot_distance"]


def shots(ev: dict[str, pd.DataFrame], aliases: dict[int, int]) -> pd.DataFrame:
    s = ev["shots"]
    out = pd.DataFrame({
        "game_id": s["gameId"], "shot_id": s["id"], "period": s["period"], "release_frame": s["startFrame"],
        "player_id": s["shooterId"], "made": s["outcome"].astype(float), "three": s["three"].astype(float),
        "shot_quality": pd.to_numeric(s["shotQuality"], errors="coerce"),
        "catch_and_shoot": s["catchAndShoot"].astype(float), "contested": s["contested"].astype(float),
        "release_time": pd.to_numeric(s["releaseTime"], errors="coerce"),
        "shot_distance": pd.to_numeric(s["distance"], errors="coerce"), "shot_type": s["shotType"],
    }).dropna(subset=["player_id", "release_frame"])
    out = out[out["shot_type"] != "heave"]
    out["player_id"] = out["player_id"].astype(int).map(lambda p: aliases.get(p, p))
    return out.reset_index(drop=True)


def load_before(sh: pd.DataFrame, df: pd.DataFrame, ceiling: dict[int, float]) -> pd.DataFrame:
    """Shooter load in each window before the release, running clock only (one game)."""
    sh = sh.copy()
    for c in LOAD_COLS:
        sh[c] = np.nan
    brake = -np.nan_to_num(df["a"].to_numpy(), nan=0.0)
    seg = df["seg"].to_numpy()
    ceil = df["player_id"].map(ceiling).to_numpy(dtype=float)
    hard = run_starts(brake >= DECEL_PCT * ceil, seg, round(EFFORT_MIN_S * FPS))
    run = df.assign(hard=hard)[df["running"]]
    groups = run.groupby("player_id").indices
    fr, gc, per = run["frame"].to_numpy(), run["game_clock"].to_numpy(), run["period"].to_numpy()
    v, hd = run["v"].to_numpy(), run["hard"].to_numpy()
    for i, r in enumerate(sh.itertuples(index=False)):
        idx = groups.get(r.player_id)
        if idx is None:
            continue
        end = np.searchsorted(fr[idx], r.release_frame)
        for w in WINDOWS_S:
            n = w * FPS
            if end < n:
                continue
            sl = idx[end - n:end]
            if per[sl[0]] != r.period or per[sl[-1]] != r.period or gc[sl[0]] - gc[sl[-1]] > w + SPAN_TOL_S:
                continue
            sh.loc[i, f"dist_per_min_{w}s"] = v[sl].sum() / FPS * 60 / w
            sh.loc[i, f"decels_per_min_{w}s"] = hd[sl].sum() * 60 / w
    return sh


def add_expected(sh: pd.DataFrame) -> pd.DataFrame:
    """Make probability from shot quality and 2 vs 3 (logistic fit on all scored shots)."""
    sh = sh.copy()
    ok = sh[["shot_quality", "three", "made"]].notna().all(axis=1)
    X = np.column_stack([np.ones(ok.sum()), sh.loc[ok, "shot_quality"] / 100, sh.loc[ok, "three"]])
    y = sh.loc[ok, "made"].to_numpy()
    beta = np.zeros(3)
    for _ in range(50):                       # Newton steps
        p = 1 / (1 + np.exp(-X @ beta))
        beta += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    sh["expected_make"] = np.nan
    sh.loc[ok, "expected_make"] = 1 / (1 + np.exp(-X @ beta))
    sh["make_vs_expected"] = sh["made"] - sh["expected_make"]
    return sh


def rmcorr(d: pd.DataFrame, x: str, y: str, subject: str = "subject") -> dict:
    """Repeated-measures correlation: Pearson r of subject-centered x and y, df = N - k - 1."""
    d = d[[subject, x, y]].dropna()
    d = d[d.groupby(subject)[x].transform("size") >= 2]
    k = d[subject].nunique()
    xc = d[x] - d.groupby(subject)[x].transform("mean")
    yc = d[y] - d.groupby(subject)[y].transform("mean")
    dof = len(d) - k - 1
    if dof < 3 or xc.std() == 0 or yc.std() == 0:
        return {"x": x, "y": y, "n": len(d), "subjects": k, "r": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    r = float(np.corrcoef(xc, yc)[0, 1])
    t = r * np.sqrt(dof / (1 - r ** 2))
    z, se = np.arctanh(r), 1 / np.sqrt(dof - 1)
    return {"x": x, "y": y, "n": len(d), "subjects": k, "r": r,
            "lo": float(np.tanh(z - 1.96 * se)), "hi": float(np.tanh(z + 1.96 * se)),
            "p": float(2 * stats.t.sf(abs(t), dof))}


def correlations(sh: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([rmcorr(sh, x, y) for x in LOAD_COLS for y in OUTCOMES])


def load_bins(sh: pd.DataFrame, x: str, n_bins: int = 5, n_boot: int = 1000, seed: int = 11) -> pd.DataFrame:
    """Outcome means across bins of within-player-game load (load minus the subject's mean)."""
    d = sh.dropna(subset=[x]).copy()
    d = d[d.groupby("subject")[x].transform("size") >= 2]
    d["centered"] = d[x] - d.groupby("subject")[x].transform("mean")
    d["bin"] = pd.qcut(d["centered"], n_bins, labels=range(1, n_bins + 1)).astype(int)
    cols = ["make_vs_expected", "shot_quality", "contested", "three"]
    point = d.groupby("bin")[cols].mean()
    rng = np.random.default_rng(seed)
    games = d["game_id"].unique()
    by_game = {g: t for g, t in d.groupby("game_id")}
    boots = pd.concat([pd.concat([by_game[g] for g in rng.choice(games, len(games))]).groupby("bin")[cols].mean()
                       for _ in range(n_boot)], keys=range(n_boot))
    lo = boots.groupby(level=1).quantile(0.025).add_suffix("_lo")
    hi = boots.groupby(level=1).quantile(0.975).add_suffix("_hi")
    out = point.join(lo).join(hi)
    out.insert(0, "n", d.groupby("bin").size())
    out.insert(1, "centered_load", d.groupby("bin")["centered"].mean())
    return out.reset_index().assign(load=x)


ALERT_Q = 0.80         # flag: load above the player's own 80th percentile of pre-shot load
ALERT_MIN_SHOTS = 5


def alert_yield(sh: pd.DataFrame, x: str) -> pd.DataFrame:
    """Alert Yield for a pre-shot fatigue flag, as in WAIMS: does a flag precede worse shooting?

    Flag = load above the player's own ALERT_Q quantile across their shots (players with
    ALERT_MIN_SHOTS+ valid shots). For flagged and unflagged shots: misses against the misses
    expected from shot quality (miss_ratio over 1 = the flag preceded worse shooting), and
    the false positive share (flagged shots that were made). AUC: player-centered load as a
    predictor of a miss (0.5 = no signal).
    """
    d = sh.dropna(subset=[x, "expected_make"]).copy()
    d = d[d.groupby("player_id")[x].transform("size") >= ALERT_MIN_SHOTS]
    d["flag"] = d[x] > d.groupby("player_id")[x].transform(lambda s: s.quantile(ALERT_Q))
    g = d.groupby("flag").agg(shots=("made", "size"), made=("made", "sum"), expected=("expected_make", "sum"))
    g["misses"] = g["shots"] - g["made"]
    g["expected_misses"] = g["shots"] - g["expected"]
    g["miss_ratio"] = g["misses"] / g["expected_misses"]       # over 1 = more misses than shot quality predicts
    g["make_vs_expected_pct"] = (g["made"] - g["expected"]) / g["shots"] * 100
    g["false_positive_share"] = np.where(g.index, g["made"] / g["shots"], np.nan)
    c = d[x] - d.groupby("player_id")[x].transform("mean")
    miss = d["made"] == 0
    auc = stats.mannwhitneyu(c[miss], c[~miss]).statistic / (miss.sum() * (~miss).sum())
    return g.reset_index().assign(load=x, players=d["player_id"].nunique(), auc_load_for_miss=auc)


def release_by_period(sh: pd.DataFrame, n_boot: int = 1000, seed: int = 11) -> pd.DataFrame:
    """Time to get the shot off (releaseTime) by quarter, catch-and-shoot jumpers and all jumpers."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, m in (("catch_and_shoot_jumpers", (sh["shot_type"] == "jumper") & (sh["catch_and_shoot"] == 1)),
                     ("all_jumpers", sh["shot_type"] == "jumper")):
        d = sh[m & (sh["period"] <= 4)].dropna(subset=["release_time"])
        games = d["game_id"].unique()
        by_game = {g: t for g, t in d.groupby("game_id")}
        point = d.groupby("period")["release_time"].agg(["size", "mean"])
        boots = pd.concat([pd.concat([by_game[g] for g in rng.choice(games, len(games))])
                           .groupby("period")["release_time"].mean() for _ in range(n_boot)], axis=1)
        point["lo"], point["hi"] = boots.quantile(0.025, axis=1), boots.quantile(0.975, axis=1)
        rows.append(point.rename(columns={"size": "n", "mean": "release_time_s"}).reset_index().assign(shots=label))
    return pd.concat(rows, ignore_index=True)


def release_correlations(sh: pd.DataFrame) -> pd.DataFrame:
    """Within player-game: does release time grow by quarter, and does a slower release cost makes?"""
    j = sh[sh["shot_type"] == "jumper"]
    cs = j[j["catch_and_shoot"] == 1]
    return pd.DataFrame([
        rmcorr(j, "period", "release_time") | {"shots": "all_jumpers"},
        rmcorr(cs, "period", "release_time") | {"shots": "catch_and_shoot_jumpers"},
        rmcorr(j, "release_time", "make_vs_expected") | {"shots": "all_jumpers"},
        rmcorr(cs, "release_time", "make_vs_expected") | {"shots": "catch_and_shoot_jumpers"},
    ])
