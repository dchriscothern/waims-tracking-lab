"""Load SkillCorner ACB open data: game metadata, Dynamic Events and tracking.

All coordinates are feet, origin at court center. Speeds are ft/s. 25 fps.
Data: SkillCorner Open Data (MIT). https://github.com/SkillCorner/opendata-basketball
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "data"
INTERIM = ROOT / "data" / "interim"
FPS = 25


def match_dir(game_id: int) -> Path:
    return RAW / "matches" / str(game_id)


def load_matches() -> pd.DataFrame:
    """One row per game from matches.json (preferred source for dates)."""
    rows = json.loads((RAW / "matches.json").read_text(encoding="utf-8"))
    return pd.DataFrame(
        {
            "game_id": [int(r["id"]) for r in rows],
            "game_date": [r["date_time"][:10] for r in rows],
            "home_team": [r["home_team"]["name"] for r in rows],
            "away_team": [r["away_team"]["name"] for r in rows],
            "home_score": [r["home_score"] for r in rows],
            "away_score": [r["away_score"] for r in rows],
        }
    )


def load_aliases() -> dict[int, int]:
    """Map duplicate player ids to one canonical id."""
    df = pd.read_csv(RAW / "player_id_aliases.csv")
    return dict(zip(df["player_id"].astype(int), df["canonical_player_id"].astype(int)))


def load_game_data(game_id: int) -> dict:
    return json.loads((match_dir(game_id) / f"{game_id}_game_data.json").read_text(encoding="utf-8"))


def load_roster(game_id: int) -> pd.DataFrame:
    """One row per rostered player with team and side."""
    gd = load_game_data(game_id)
    rows = []
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        team = gd[key]
        for p in team["players"]:
            rows.append(
                {
                    "game_id": game_id,
                    "side": side,
                    "team_id": team["teamId"],
                    "team_name": team["teamName"],
                    "player_id": int(p["playerId"]),
                    "player_name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                    "jersey": p.get("jersey"),
                }
            )
    return pd.DataFrame(rows)


def load_events(game_id: int) -> dict[str, pd.DataFrame]:
    """All 20 Dynamic Events tables for one game, keyed by table name."""
    raw = json.loads((match_dir(game_id) / f"{game_id}_dynamic_events.json").read_text(encoding="utf-8"))
    out = {k: pd.DataFrame(v) for k, v in raw.items()}
    if "closeouts" in out and "touchWallClock" in out["closeouts"]:
        out["closeouts"]["touchWallClock"] = pd.to_numeric(out["closeouts"]["touchWallClock"], errors="coerce")
    return out


def parse_tracking(game_id: int) -> pd.DataFrame:
    """Stream the gzip JSONL into a long player-frame table.

    Dead-time frames (empty player lists) are dropped. One row per player per frame.
    """
    path = match_dir(game_id) / f"{game_id}_tracking_data.jsonl.gz"
    cols: dict[str, list] = {k: [] for k in
                             ("frame", "period", "game_clock", "clock_stopped", "side",
                              "player_id", "x", "y", "speed", "detected", "pred_error")}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            fr = json.loads(line)
            if not fr["homePlayers"] and not fr["awayPlayers"]:
                continue
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                for p in fr[key]:
                    cols["frame"].append(fr["frameIdx"])
                    cols["period"].append(fr["period"])
                    cols["game_clock"].append(fr["gameClock"])
                    cols["clock_stopped"].append(bool(fr["gameClockStopped"]))
                    cols["side"].append(side)
                    cols["player_id"].append(int(p["playerId"]))
                    cols["x"].append(p["xyz"][0])
                    cols["y"].append(p["xyz"][1])
                    cols["speed"].append(p["speed"])
                    cols["detected"].append(bool(p["isDetected"]))
                    cols["pred_error"].append(p["predError"])
    df = pd.DataFrame(cols)
    df["game_id"] = game_id
    df = df.astype({"frame": "int32", "period": "int8", "player_id": "int32",
                    "x": "float32", "y": "float32", "speed": "float32",
                    "pred_error": "float32", "game_clock": "float32"})
    return df.sort_values(["player_id", "frame"], ignore_index=True)


def tracking(game_id: int, rebuild: bool = False) -> pd.DataFrame:
    """Cached player-frame table from data/interim/{game_id}_players.parquet."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / f"{game_id}_players.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = parse_tracking(game_id)
    df.to_parquet(path, index=False)
    return df
