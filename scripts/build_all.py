"""Run the full pipeline: raw tracking -> cleaned -> metrics -> roles -> action cost -> exports.

Usage:  python scripts/build_all.py
Outputs:
  data/interim/results/*.csv   full-detail tables (gitignored)
  exports/waims/*.csv           small tables for WAIMS (committed)
  exports/figures/*.png         figures for README and portfolio
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtl import actions, clean, io, metrics, roles, viz  # noqa: E402

RESULTS = ROOT / "data" / "interim" / "results"
WAIMS = ROOT / "exports" / "waims"
SENSITIVITY = [("savgol", 7), ("savgol", 13), ("savgol", 25), ("butter", 0), ("none", 0)]


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    WAIMS.mkdir(parents=True, exist_ok=True)
    matches = io.load_matches()
    aliases = io.load_aliases()

    pg, per, w30, w60, meas, sens, rosters, events_by_game = [], [], [], [], [], [], [], {}
    for gid in matches["game_id"]:
        raw = io.tracking(gid)
        ev = io.load_events(gid)
        events_by_game[gid] = ev
        rosters.append(io.load_roster(gid))

        df = metrics.add_flags(clean.clean(raw, aliases))
        g_pg = metrics.player_game(df)
        pg.append(g_pg)
        per.append(metrics.by_period(df))
        for secs, bucket in ((30, w30), (60, w60)):
            bucket.append(metrics.top_windows(metrics.peak_windows(df, secs), k=3, seconds=secs)
                          .assign(window_s=secs))
        baseline = g_pg.set_index("player_id")["distance_per_min"] / 60
        meas.append(actions.measure(actions.action_rows(ev, aliases), df, baseline))

        for method, win in SENSITIVITY:
            s = metrics.add_flags(clean.clean(raw, aliases, method=method, window=win or 13))
            s_pg = metrics.player_game(s)
            sens.append(s_pg.assign(method=f"{method}{'_' + str(win) if win else ''}")
                        [["game_id", "player_id", "method", "minutes_running", "distance_per_min",
                          "accels_per_min", "decels_per_min", "sprints"]])
        print(f"game {gid} done ({time.time() - t0:.0f}s)", flush=True)

    pg = pd.concat(pg, ignore_index=True)
    per = pd.concat(per, ignore_index=True)
    peaks = pd.concat(w30 + w60, ignore_index=True)
    meas = pd.concat(meas, ignore_index=True)
    sens = pd.concat(sens, ignore_index=True)
    roster = pd.concat(rosters, ignore_index=True)
    roster["player_id"] = roster["player_id"].map(lambda p: aliases.get(p, p))

    # Roles, pooled across games
    minutes = pg.groupby("player_id")["minutes_running"].sum()
    feats = roles.features(events_by_game, minutes, aliases)
    rl = roles.assign(feats)
    role_of = rl["role"]
    agree = (rl["role_rule"] == rl["role_kmeans"]).mean()
    print(f"roles: {rl['role'].value_counts().to_dict()}, k-means agreement {agree:.0%}")

    for t in (pg, per, peaks, meas, sens):
        t["role"] = t["player_id"].map(role_of)

    # Peak window context: which actions a player took part in during top 60 s windows
    top60 = peaks[(peaks["window_s"] == 60) & (peaks["rank"] == 1)]
    ctx = []
    for r in top60.itertuples(index=False):
        m = meas[(meas["game_id"] == r.game_id) & (meas["player_id"] == r.player_id)
                 & (meas["start_frame"] <= r.end_frame) & (meas["end_frame"] >= r.start_frame)]
        ctx.append(m.assign(role=r.role))
    ctx = pd.concat(ctx, ignore_index=True) if ctx else pd.DataFrame()
    per_window = ctx.groupby("role")["action_id"].size() / top60.groupby("role").size()
    ctx_mix = (ctx.groupby(["role", "action", "actor_side"]).size()
               .div(top60.groupby("role").size(), level="role")
               .rename("per_window").reset_index())
    # Same rate for all running time, per 60 s, for comparison
    all_rate = (meas.groupby(["role", "action", "actor_side"]).size()
                .div(pg.groupby("role")["minutes_running"].sum(), level="role")
                .rename("per_minute_overall").reset_index())
    ctx_mix = ctx_mix.merge(all_rate, on=["role", "action", "actor_side"], how="left")
    ctx_mix["ratio_vs_overall"] = ctx_mix["per_window"] / ctx_mix["per_minute_overall"]

    cost = actions.cost_table(meas, role_of)

    # Full-detail results
    pg.merge(roster.drop_duplicates(["game_id", "player_id"]), on=["game_id", "side", "player_id"], how="left") \
        .to_csv(RESULTS / "player_game.csv", index=False)
    per.to_csv(RESULTS / "player_period.csv", index=False)
    peaks.to_csv(RESULTS / "peak_windows.csv", index=False)
    meas.to_csv(RESULTS / "action_windows.csv", index=False)
    sens.to_csv(RESULTS / "smoothing_sensitivity.csv", index=False)
    rl.to_csv(RESULTS / "roles.csv")
    roles.summary(rl).to_csv(RESULTS / "role_summary.csv")
    ctx_mix.to_csv(RESULTS / "peak_window_context.csv", index=False)

    write_waims(pg, per, peaks, cost, matches, roster)
    write_app_extras(meas, ctx_mix)
    viz.all_figures(pg, per, peaks, meas, sens, cost, ctx_mix, rl, ROOT / "exports" / "figures")
    print(f"actions per top-60s window by role: {per_window.round(1).to_dict()}")
    print(f"done in {time.time() - t0:.0f}s")


def write_waims(pg, per, peaks, cost, matches, roster) -> None:
    """Anonymized, small CSVs that WAIMS reads. Contract in BUILD_SPEC.md section 6."""
    team = roster.drop_duplicates(["game_id", "player_id"])[["game_id", "player_id", "team_name"]]
    p = pg.merge(matches[["game_id", "game_date"]], on="game_id").merge(team, on=["game_id", "player_id"], how="left")
    labels = {pid: f"Player {i + 1:02d}" for i, pid in enumerate(sorted(p["player_id"].unique()))}
    peak60 = peaks[(peaks["window_s"] == 60) & (peaks["rank"] == 1)] \
        .set_index(["game_id", "player_id"])["distance_per_min"].rename("peak60_dist_per_min")
    peak30 = peaks[(peaks["window_s"] == 30) & (peaks["rank"] == 1)] \
        .set_index(["game_id", "player_id"])["distance_per_min"].rename("peak30_dist_per_min")
    p = p.join(peak60, on=["game_id", "player_id"]).join(peak30, on=["game_id", "player_id"])
    p["player_label"] = p["player_id"].map(labels)
    p["role"] = p["role"].fillna("unassigned")
    cols = ["game_id", "game_date", "team_name", "player_label", "role", "minutes_running",
            "distance_ft", "distance_per_min", "hsd_ft", "hsd_per_min", "sprints", "accels", "decels",
            "efforts_per_min", "peak_speed", "peak30_dist_per_min", "peak60_dist_per_min", "detected_share"]
    p[cols].round(3).to_csv(WAIMS / "game_demands_player.csv", index=False)

    # Benchmarks from player-games with meaningful minutes
    q = p[(p["minutes_running"] >= 10) & (p["role"] != "unassigned")]
    mets = ["distance_per_min", "hsd_per_min", "efforts_per_min", "peak_speed",
            "peak30_dist_per_min", "peak60_dist_per_min", "sprints", "distance_ft"]
    bench = []
    for role, g in q.groupby("role"):
        for m in mets:
            bench.append({"role": role, "metric": m, "p10": g[m].quantile(.1), "p50": g[m].median(),
                          "p90": g[m].quantile(.9), "n_player_games": len(g)})
    pd.DataFrame(bench).round(3).to_csv(WAIMS / "role_benchmarks.csv", index=False)

    per_q = per[per["minutes"] >= 3].dropna(subset=["role"]) \
        .groupby(["role", "period"])[["distance_per_min", "efforts_per_min"]].median()
    per_q.round(3).reset_index().to_csv(WAIMS / "role_by_quarter.csv", index=False)
    cost.round(3).to_csv(WAIMS / "action_cost.csv", index=False)


def write_app_extras(meas, ctx_mix) -> None:
    """Extra small tables for the Streamlit app (no player ids)."""
    p = meas[(meas["action"] == "picks") & meas["actor"].isin(["ballhandler_def", "screener_def"])].dropna(
        subset=["coverage"])
    g = p.groupby(["actor", "coverage"])["excess_speed"].agg(["mean", "std", "size"]).reset_index()
    g = g[g["size"] >= 20].rename(columns={"mean": "mean_excess_speed", "size": "n"})
    g["ci95"] = 1.96 * g.pop("std") / np.sqrt(g["n"])
    g.round(3).to_csv(WAIMS / "pick_coverage.csv", index=False)
    ctx_mix.round(3).to_csv(WAIMS / "peak_context.csv", index=False)
    d = meas[meas["action"] == "drives"].groupby(["actor", "blowby"])["excess_speed"].agg(["mean", "size"])
    d.rename(columns={"mean": "mean_excess_speed", "size": "n"}).round(3).reset_index() \
        .to_csv(WAIMS / "drive_blowby.csv", index=False)


if __name__ == "__main__":
    main()
