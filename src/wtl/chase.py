"""Chase Cost vs gravity (pick 3).

Chase Cost: how much running an offensive player forces on whoever guards them.
For every matchup interval (matchups table: defender assigned to offensive player),
the defender's distance and hard decels per running-clock minute, minus that
defender's own average while defending in the same game ("excess"). An offensive
player's Chase Cost is the time-weighted mean excess of their defenders.

Gravity (simple proxy, context only): how tightly a player is guarded off the ball,
compared with what their court position predicts. The defender gap is regressed on
the player's distance to the hoop and to the ball (with squares); gravity is minus the
player's mean residual, in feet (positive = guarded tighter than position predicts).

Tracking coordinates are not normalized to one attacking direction, so the hoop
side of each interval is taken from the mean x of the ten players in it.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from .io import FPS, INTERIM, match_dir
from .metrics import EFFORT_MIN_S, run_starts

HOOP_X = 40.75          # ft from center (fitted from shot distances; FIBA court)
DECEL_PCT = 0.75
SAMPLE_EVERY = 5        # frames, for the gravity regression
MIN_GUARDED_MIN = 5.0   # minutes guarded per player-game to report Chase Cost


def ball(game_id: int) -> pd.DataFrame:
    """Ball x, y by frame (cached). Ball position is always estimated in this data."""
    path = INTERIM / f"{game_id}_ball.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    with gzip.open(match_dir(game_id) / f"{game_id}_tracking_data.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            fr = json.loads(line)
            b = fr.get("ball") or {}
            if b.get("xyz") and (fr["homePlayers"] or fr["awayPlayers"]):
                rows.append((fr["frameIdx"], b["xyz"][0], b["xyz"][1]))
    out = pd.DataFrame(rows, columns=["frame", "bx", "by"]).astype({"frame": "int32", "bx": "float32", "by": "float32"})
    out.to_parquet(path, index=False)
    return out


def _in_touch(frames: np.ndarray, spans: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """True where a frame falls inside one of the player's touches (sorted start, end arrays)."""
    if spans is None:
        return np.zeros(len(frames), dtype=bool)
    s, e = spans
    k = np.searchsorted(s, frames, side="right") - 1
    return (k >= 0) & (frames <= e[np.clip(k, 0, None)])


def game(df: pd.DataFrame, ev: dict[str, pd.DataFrame], bl: pd.DataFrame, ceiling: dict[int, float],
         aliases: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per matchup interval defender load, and off-ball gravity samples, for one game."""
    brake = -np.nan_to_num(df["a"].to_numpy(), nan=0.0)
    ceil = df["player_id"].map(ceiling).to_numpy(dtype=float)
    df = df.assign(hard=run_starts(brake >= DECEL_PCT * ceil, df["seg"].to_numpy(), round(EFFORT_MIN_S * FPS)))
    run = df[df["running"]]
    idx = run.groupby("player_id").indices
    fr, x, y = run["frame"].to_numpy(), run["x"].to_numpy(), run["y"].to_numpy()
    v, hard = run["v"].to_numpy(), run["hard"].to_numpy()
    bxy = bl.set_index("frame")[["bx", "by"]]

    tou = ev["touches"].dropna(subset=["playerId"])
    touch = {}
    for pid, t in tou.assign(pid=tou["playerId"].astype(int).map(lambda p: aliases.get(p, p))).groupby("pid"):
        t = t.sort_values("startFrame")
        touch[pid] = (t["startFrame"].to_numpy(), t["endFrame"].to_numpy())

    mu = ev["matchups"].dropna(subset=["defPlayerId", "offPlayerId", "startFrame", "endFrame"])
    rows, samples = [], []
    for r in mu.itertuples(index=False):
        d, o = aliases.get(int(r.defPlayerId), int(r.defPlayerId)), aliases.get(int(r.offPlayerId), int(r.offPlayerId))
        di, oi = idx.get(d), idx.get(o)
        if di is None or oi is None:
            continue
        ds = di[np.searchsorted(fr[di], r.startFrame):np.searchsorted(fr[di], r.endFrame, side="right")]
        os_ = oi[np.searchsorted(fr[oi], r.startFrame):np.searchsorted(fr[oi], r.endFrame, side="right")]
        if len(ds) < FPS or len(os_) < FPS:
            continue
        row = {"game_id": r.gameId, "chance_id": r.chanceId, "def_id": d, "off_id": o,
               "seconds": len(ds) / FPS, "def_dist_ft": v[ds].sum() / FPS, "def_hard": int(hard[ds].sum())}
        on_d = _in_touch(fr[ds], touch.get(o))
        for state, m in (("on", on_d), ("off", ~on_d)):
            row[f"seconds_{state}"] = m.sum() / FPS
            row[f"def_dist_ft_{state}"] = v[ds][m].sum() / FPS
            row[f"def_hard_{state}"] = int(hard[ds][m].sum())
        rows.append(row)
        # Gravity samples: off-ball frames where both are tracked
        common, oj, dj = np.intersect1d(fr[os_], fr[ds], return_indices=True)
        keep = np.arange(0, len(common), SAMPLE_EVERY)
        common, oj, dj = common[keep], os_[oj[keep]], ds[dj[keep]]
        on = _in_touch(common, touch.get(o))
        common, oj, dj = common[~on], oj[~on], dj[~on]
        if not len(common):
            continue
        b = bxy.reindex(common).to_numpy()
        hoop = np.sign(np.nanmean(np.r_[x[oj], x[dj]])) * HOOP_X
        samples.append(pd.DataFrame({
            "game_id": r.gameId, "off_id": o,
            "gap": np.hypot(x[oj] - x[dj], y[oj] - y[dj]),
            "hoop_dist": np.hypot(x[oj] - hoop, y[oj]),
            "ball_dist": np.hypot(x[oj] - b[:, 0], y[oj] - b[:, 1]),
        }))
    return pd.DataFrame(rows), (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame())


def chase_cost(rows: pd.DataFrame) -> pd.DataFrame:
    """Per offensive player-game: their defenders' excess distance and hard decels per minute."""
    r = rows.copy()
    base = r.groupby(["game_id", "def_id"])[["def_dist_ft", "def_hard", "seconds"]].transform("sum")
    r["excess_ft"] = r["def_dist_ft"] - base["def_dist_ft"] / base["seconds"] * r["seconds"]
    r["excess_hard"] = r["def_hard"] - base["def_hard"] / base["seconds"] * r["seconds"]
    g = r.groupby(["game_id", "off_id"])[["excess_ft", "excess_hard", "seconds", "def_dist_ft"]].sum()
    out = pd.DataFrame({
        "minutes_guarded": g["seconds"] / 60,
        "chase_cost_ft_per_min": g["excess_ft"] / g["seconds"] * 60,
        "chase_cost_decels_per_min": g["excess_hard"] / g["seconds"] * 60,
        "defender_ft_per_min": g["def_dist_ft"] / g["seconds"] * 60,
    })
    # Same, split by whether the offensive player had the ball; each state against the defender's own
    # average in that state
    for st in ("on", "off"):
        sec, ft, hd = f"seconds_{st}", f"def_dist_ft_{st}", f"def_hard_{st}"
        b = r.groupby(["game_id", "def_id"])[[ft, hd, sec]].transform("sum")
        rate_ft = (b[ft] / b[sec]).where(b[sec] > 0)
        rate_hd = (b[hd] / b[sec]).where(b[sec] > 0)
        s = r.assign(xf=r[ft] - rate_ft * r[sec], xh=r[hd] - rate_hd * r[sec]) \
            .groupby(["game_id", "off_id"])[["xf", "xh", sec]].sum()
        out[f"minutes_{st}_ball"] = s[sec] / 60
        out[f"chase_cost_{st}_ball_ft_per_min"] = (s["xf"] / s[sec] * 60).where(s[sec] > 0)
        out[f"chase_cost_{st}_ball_decels_per_min"] = (s["xh"] / s[sec] * 60).where(s[sec] > 0)
    return out.reset_index()


def gravity(samples: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per offensive player-game gravity (ft tighter than position predicts) and the model's R^2."""
    s = samples.dropna()
    s = s[(s["hoop_dist"] < 30) & (s["gap"] < 30)]            # half-court, a real matchup
    X = np.column_stack([np.ones(len(s)), s["hoop_dist"], s["ball_dist"], s["hoop_dist"] ** 2, s["ball_dist"] ** 2])
    beta, *_ = np.linalg.lstsq(X, s["gap"].to_numpy(), rcond=None)
    resid = s["gap"].to_numpy() - X @ beta
    r2 = 1 - resid.var() / s["gap"].var()
    g = s.assign(resid=resid).groupby(["game_id", "off_id"]).agg(
        gravity_ft=("resid", lambda z: -z.mean()), gravity_samples=("resid", "size"),
        mean_gap_ft=("gap", "mean")).reset_index()
    return g, float(r2)


def reliability(pg: pd.DataFrame, col: str) -> dict:
    """Game-to-game stability: each player's first two qualifying games, Pearson and Spearman."""
    d = pg.dropna(subset=[col]).sort_values(["player_id", "game_id"])
    two = d.groupby("player_id").head(2)
    w = two.assign(k=two.groupby("player_id").cumcount()).pivot(index="player_id", columns="k", values=col).dropna()
    if len(w) < 5:
        return {"metric": col, "players": len(w), "pearson": np.nan, "spearman": np.nan}
    return {"metric": col, "players": len(w), "pearson": w[0].corr(w[1]),
            "spearman": w[0].corr(w[1], method="spearman")}


def role_table(pg: pd.DataFrame, n_boot: int = 1000, seed: int = 11) -> pd.DataFrame:
    """Chase Cost by the role of the player being guarded, on and off the ball.

    Means weighted by minutes in that state, with 95% intervals from a bootstrap that resamples games.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for st in ("on", "off"):
        c, w = f"chase_cost_{st}_ball_ft_per_min", f"minutes_{st}_ball"
        d = pg.dropna(subset=[c, "role"])
        d = d[d[w] > 0]

        def mean(t: pd.DataFrame) -> pd.Series:
            return t.groupby("role").apply(lambda r: np.average(r[c], weights=r[w]), include_groups=False)

        games = d["game_id"].unique()
        by_game = {g: t for g, t in d.groupby("game_id")}
        boots = pd.concat([mean(pd.concat([by_game[g] for g in rng.choice(games, len(games))]))
                           for _ in range(n_boot)], axis=1)
        rows.append(pd.DataFrame({"ball": st, "chase_cost_ft_per_min": mean(d),
                                  "lo": boots.quantile(0.025, axis=1), "hi": boots.quantile(0.975, axis=1),
                                  "minutes": d.groupby("role")[w].sum(),
                                  "player_games": d.groupby("role").size()}).rename_axis("role").reset_index())
    return pd.concat(rows, ignore_index=True)
