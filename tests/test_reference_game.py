"""Regression checks against values verified by hand on game 114243 (Manresa vs Granada).

Needs the raw data: run `python scripts/download_data.py 114243` first.
Unit tests at the bottom run without any data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtl import actions, clean, io, metrics  # noqa: E402

GAME = 114243
HAS_DATA = (io.match_dir(GAME) / f"{GAME}_tracking_data.jsonl.gz").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="raw data not downloaded")


@pytest.fixture(scope="module")
def game():
    df = metrics.add_flags(clean.clean(io.tracking(GAME), io.load_aliases()))
    return df, metrics.player_game(df), io.load_events(GAME)


@needs_data
def test_frame_counts(game):
    df, _, _ = game
    assert df["frame"].nunique() == 77_023
    assert df.groupby("frame").size().eq(10).all()
    assert df["detected"].mean() == pytest.approx(0.826, abs=0.002)


@needs_data
def test_running_minutes_match_game_length(game):
    _, pg, _ = game
    team = pg.groupby("side")["minutes_running"].sum()
    assert team.between(196, 200).all()


@needs_data
def test_minutes_cross_check_with_chance_players(game):
    _, pg, ev = game
    top = pg.sort_values("minutes_running").iloc[-1]
    sk = ev["chance_players"].groupby("playerId")["minutesPlayedBefore"].max()
    assert top["minutes_running"] == pytest.approx(sk[top["player_id"]], abs=0.5)


@needs_data
def test_top_player_distance(game):
    _, pg, _ = game
    top = pg.sort_values("minutes_running").iloc[-1]
    assert top["distance_per_min"] == pytest.approx(411.6, rel=0.01)


@needs_data
def test_speed_is_capped_and_smooth(game):
    df, _, _ = game
    assert df["v"].max() <= clean.SPEED_CAP
    assert df["capped"].sum() < 100


@needs_data
def test_drive_join_face_validity(game):
    df, pg, ev = game
    base = pg.set_index("player_id")["distance_per_min"] / 60
    m = actions.measure(actions.action_rows(ev, io.load_aliases()), df, base)
    d = m[m["action"] == "drives"]
    handler = d[d["actor"] == "ballhandler"]["mean_speed"].mean()
    defender = d[d["actor"] == "ballhandler_def"]["mean_speed"].mean()
    assert handler == pytest.approx(11.6, abs=0.3)
    assert handler > defender


# Unit tests (no data needed)

def test_run_starts_min_length_and_segments():
    cond = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    seg = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = metrics.run_starts(cond, seg, 3)
    # run at 0 (len 3) counts, run at 4 is split by segment (len 2 + 2), run at 9 too short
    assert out.tolist() == [True] + [False] * 10


def test_top_windows_do_not_overlap():
    w = pd.DataFrame({"game_id": 1, "side": "home", "player_id": 7, "period": 1,
                      "start_frame": [0, 10, 5000], "end_frame": [2000, 2010, 6600],
                      "distance_per_min": [500, 499, 450], "efforts": [1, 1, 1]})
    t = metrics.top_windows(w, k=3)
    assert t["start_frame"].tolist() == [0, 5000]
