"""Load before a shot vs shot execution (pick 2).

Usage:  python scripts/fatigue.py
Outputs:
  data/interim/results/shots_load.csv          one row per shot with load before it (gitignored)
  exports/fatigue/fatigue_rmcorr.csv           within-player-game correlations, load vs outcome
  exports/fatigue/fatigue_load_bins.csv        outcomes across 5 within-player load bins
  exports/fatigue/fatigue_alert_yield.csv      flagged (own p80 load) vs unflagged shots, makes vs expected
  exports/fatigue/release_by_period.csv        time to get the shot off by quarter
  exports/fatigue/release_rmcorr.csv           release time vs quarter and vs make vs expected

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

from wtl import clean, closeouts, fatigue, io  # noqa: E402

RESULTS = ROOT / "data" / "interim" / "results"
OUT = ROOT / "exports" / "fatigue"


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    aliases = io.load_aliases()
    games = io.load_matches()["game_id"].tolist()

    # Pass 1: each player's braking ceiling, pooled across games (same definition as closeouts)
    braking = {}
    for gid in games:
        for pid, arr in closeouts.decel_ceiling(clean.clean(io.tracking(gid), aliases)).items():
            braking.setdefault(pid, []).append(arr)
    ceiling = {pid: np.quantile(np.concatenate(a), closeouts.DECEL_CEIL_Q) for pid, a in braking.items()}
    print(f"ceilings done ({time.time() - t0:.0f}s)", flush=True)

    # Pass 2: load before each shot
    out = []
    for gid in games:
        df = clean.clean(io.tracking(gid), aliases)
        out.append(fatigue.load_before(fatigue.shots(io.load_events(gid), aliases), df, ceiling))
        print(f"game {gid} ({time.time() - t0:.0f}s)", flush=True)
    sh = fatigue.add_expected(pd.concat(out, ignore_index=True))
    sh["subject"] = sh["game_id"].astype(str) + "_" + sh["player_id"].astype(str)
    sh.to_csv(RESULTS / "shots_load.csv", index=False)

    fatigue.correlations(sh).round(4).to_csv(OUT / "fatigue_rmcorr.csv", index=False)
    pd.concat([fatigue.load_bins(sh, c) for c in fatigue.LOAD_COLS]).round(4) \
        .to_csv(OUT / "fatigue_load_bins.csv", index=False)
    pd.concat([fatigue.alert_yield(sh, c) for c in fatigue.LOAD_COLS]).round(4) \
        .to_csv(OUT / "fatigue_alert_yield.csv", index=False)
    fatigue.release_by_period(sh).round(4).to_csv(OUT / "release_by_period.csv", index=False)
    fatigue.release_correlations(sh).round(4).to_csv(OUT / "release_rmcorr.csv", index=False)
    cover = sh[fatigue.LOAD_COLS].notna().mean().round(2).to_dict()
    print(f"{len(sh)} shots; share with a valid window: {cover}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
