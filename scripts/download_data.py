"""Download SkillCorner ACB open data (metadata, events, aggregates, tracking).

Non-tracking files come from raw.githubusercontent.com. Tracking files are Git LFS
objects, so they come from media.githubusercontent.com. Existing files are skipped.
Data: SkillCorner Open Data (MIT). https://github.com/SkillCorner/opendata-basketball
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPO = "SkillCorner/opendata-basketball"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/"
LFS_URL = f"https://media.githubusercontent.com/media/{REPO}/main/"

SMALL_FILES = [
    "data/matches.json",
    "data/player_id_aliases.csv",
    "data/aggregates/acb_shotsaggregates_20252026.csv",
    "data/aggregates/acb_drivesaggregates_20252026.csv",
    "data/aggregates/acb_picksaggregates_20252026.csv",
]


def fetch(url: str, dest: Path, min_bytes: int = 0) -> None:
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f"skip   {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch  {dest.relative_to(ROOT)}", flush=True)
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(url, tmp)
            break
        except (urllib.error.ContentTooShortError, urllib.error.URLError, OSError) as err:
            if attempt == 3:
                raise
            print(f"       retry {attempt} after: {err}", flush=True)
    size = tmp.stat().st_size
    if size <= min_bytes:
        tmp.unlink()
        raise RuntimeError(f"{dest.name} is only {size} bytes (LFS pointer stub?)")
    tmp.replace(dest)


def game_ids() -> list[int]:
    matches = json.loads((RAW / "data" / "matches.json").read_text(encoding="utf-8"))
    return [int(m["id"]) for m in matches]


def main(only: list[int] | None = None) -> None:
    for rel in SMALL_FILES:
        fetch(RAW_URL + rel, RAW / rel)
    ids = only or game_ids()
    for gid in ids:
        base = f"data/matches/{gid}/{gid}"
        fetch(RAW_URL + f"{base}_game_data.json", RAW / f"{base}_game_data.json")
        fetch(RAW_URL + f"{base}_dynamic_events.json", RAW / f"{base}_dynamic_events.json")
        fetch(LFS_URL + f"{base}_tracking_data.jsonl.gz",
              RAW / f"{base}_tracking_data.jsonl.gz", min_bytes=1_000_000)
    print(f"done: {len(ids)} games in {RAW}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)
