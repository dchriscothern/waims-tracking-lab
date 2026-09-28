"""Render a player's peak-minute replay as an MP4.

Usage:
  python scripts/render_clip.py                         # best guard peak minute (auto-picked)
  python scripts/render_clip.py --game 188630 --player 59161
  python scripts/render_clip.py --role wing             # best wing peak minute
  python scripts/render_clip.py --stills 80 800 1400    # PNG stills at those video frames, no MP4

Needs data/interim/results from scripts/build_all.py.
Auto-pick: the role's top peak-60 s windows that are fully live (no stoppages),
ranked by intensity plus variety of actions on both ends of the floor.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtl import clean, io, metrics, replay  # noqa: E402

RESULTS = ROOT / "data" / "interim" / "results"
OUT = ROOT / "exports" / "video"


def pick_window(role: str, game: int | None, player: int | None) -> pd.Series:
    pk = pd.read_csv(RESULTS / "peak_windows.csv")
    acts = pd.read_csv(RESULTS / "action_windows.csv")
    w = pk[pk["window_s"] == 60].copy()
    if game and player:
        w = w[(w["game_id"] == game) & (w["player_id"] == player)].sort_values("rank")
        return w.iloc[0]
    w = w[(w["role"] == role) & (w["end_frame"] - w["start_frame"] + 1 <= 1525)]

    def score(r):
        a = acts[(acts["game_id"] == r.game_id) & (acts["player_id"] == r.player_id)
                 & (acts["start_frame"] <= r.end_frame) & (acts["end_frame"] >= r.start_frame)]
        both_ends = a["actor"].str.endswith("_def").any() and (~a["actor"].str.endswith("_def")).any()
        return r.distance_per_min / 10 + 2 * a["action"].nunique() + r.efforts + (5 if both_ends else 0)

    w["score"] = w.apply(score, axis=1)
    return w.sort_values("score", ascending=False).iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="guard")
    ap.add_argument("--game", type=int)
    ap.add_argument("--player", type=int)
    ap.add_argument("--stills", type=int, nargs="*")
    args = ap.parse_args()

    win = pick_window(args.role, args.game, args.player)
    gid, pid = int(win["game_id"]), int(win["player_id"])
    s, e = int(win["start_frame"]), int(win["end_frame"])

    pg = pd.read_csv(RESULTS / "player_game.csv")
    me = pg[(pg["game_id"] == gid) & (pg["player_id"] == pid)].iloc[0]
    m = io.load_matches().set_index("game_id").loc[gid]
    opp = m["away_team"] if me["team_name"] == m["home_team"] else m["home_team"]
    role = me["role"] if isinstance(me["role"], str) else "player"

    aliases = io.load_aliases()
    raw = io.tracking(gid)
    players = raw[(raw["frame"] >= s) & (raw["frame"] <= e)]
    full = metrics.add_flags(clean.clean(raw, aliases))
    focus = full[(full["player_id"] == pid) & (full["frame"] >= s) & (full["frame"] <= e)]
    acts = pd.read_csv(RESULTS / "action_windows.csv")
    acts = acts[(acts["game_id"] == gid) & (acts["player_id"] == pid)
                & (acts["start_frame"] <= e) & (acts["end_frame"] >= s)]

    clip = replay.Clip(
        game_id=gid, player_id=pid, start_frame=s, end_frame=e,
        title=f"Peak minute: {role}, {me['team_name']}",
        subtitle=f"vs {opp}  |  {m['game_date']}  |  Q{int(win['period'])}",
        role=role, period=int(win["period"]),
        game_avg_pace=float(me["distance_per_min"]),
        game_avg_efforts=float(me["efforts_per_min"]),
    )
    out = OUT / f"peak_minute_{role}_{gid}_{pid}.mp4"
    t0 = time.time()
    replay.render(clip, players, focus, acts, out, stills=args.stills)
    print(f"{'stills' if args.stills is not None else 'video'} -> {out.parent}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
