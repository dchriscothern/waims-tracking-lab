"""Closeouts: early vs late vs none, cost to the defender and payoff (pick 1).

Usage:  python scripts/closeouts.py
Outputs:
  data/interim/results/closeout_catches.csv   one row per perimeter catch (gitignored)
  exports/closeouts/closeout_summary.csv      group means with game-bootstrap 95% intervals
  exports/closeouts/closeout_correlations.csv Spearman's rho among closeouts
  exports/closeouts/closeout_by_shooter.csv   groups within thirds of the catcher's season 3PA per game
  exports/closeouts/closeout_quarter_drift.csv Q4 minus Q1 for the same defender and game
  exports/closeouts/closeouts.png, early_closeouts.png

Self-contained: not part of the README findings, the WAIMS feed or build_all.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtl import clean, closeouts, io  # noqa: E402

RESULTS = ROOT / "data" / "interim" / "results"
OUT = ROOT / "exports" / "closeouts"


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    aliases = io.load_aliases()
    cats, trs, braking = [], [], {}
    for gid in io.load_matches()["game_id"]:
        df = clean.clean(io.tracking(gid), aliases)
        cat = closeouts.catches(io.load_events(gid), aliases)
        cats.append(cat)
        trs.extend(closeouts.traces(cat, df))
        for pid, arr in closeouts.decel_ceiling(df).items():
            braking.setdefault(pid, []).append(arr)
        print(f"game {gid}: {len(cat)} perimeter catches ({time.time() - t0:.0f}s)", flush=True)

    cat = pd.concat(cats, ignore_index=True)
    ceiling = {pid: np.quantile(np.concatenate(a), closeouts.DECEL_CEIL_Q) for pid, a in braking.items()}
    cost = pd.DataFrame([closeouts.cost(t, ceiling.get(pid, np.nan))
                         for t, pid in zip(trs, cat["defender_id"])])
    cat = closeouts.add_splits(pd.concat([cat, cost], axis=1).dropna(subset=["peak_speed"]))
    cat["defender_ceiling"] = cat["defender_id"].map(ceiling)
    cat = closeouts.add_shooter_tier(cat, closeouts.shooter_volume(aliases))
    cat.to_csv(RESULTS / "closeout_catches.csv", index=False)
    closeouts.shooter_check(cat).round(3).to_csv(OUT / "closeout_by_shooter.csv", index=False)

    co_q = cat[(cat["group"] == "closeout") & (cat["period"] <= 4)]
    summ = pd.concat([closeouts.summary(cat, "dist_split"), closeouts.summary(cat, "lead_split"),
                      closeouts.summary(co_q, "period")])
    closeouts.quarter_drift(cat).round(3).to_csv(OUT / "closeout_quarter_drift.csv", index=False)
    summ.round(3).to_csv(OUT / "closeout_summary.csv", index=False)
    corr = closeouts.correlations(cat)
    corr.round(4).to_csv(OUT / "closeout_correlations.csv", index=False)
    closeouts.figure(summ, OUT)
    closeouts.early_figure(summ, OUT)

    print(cat["group"].value_counts().to_dict(), f"ceiling median {np.median(list(ceiling.values())):.1f} ft/s^2")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
