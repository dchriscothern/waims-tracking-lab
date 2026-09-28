# CLAUDE.md: waims-tracking-lab

## What this repo is

This repo analyzes game physical demands and the "cost of actions" using SkillCorner's open basketball data: Liga ACB 2025/26, 10 games of 25 fps broadcast tracking plus Dynamic Events.

It serves two purposes:

1. **WAIMS feed.** Small exported tables in `exports/waims/` that WAIMS Python loads for a "Game Demands (real pro data)" view. WAIMS lives in its own repo and is never edited from here.
2. **Portfolio case study** for chriscothern.com, in the Load and Monitoring category.

`README.md` holds the findings and methods. `BUILD_SPEC.md` holds the design decisions, current status and the backlog. Read both before starting a task.

## HOW TO APPROACH EVERY TASK

(Replace this section with the standard block used across the other repos if it differs.)

1. Read `BUILD_SPEC.md` and find the backlog item you are on.
2. State the plan in a few lines before editing files.
3. Make the smallest change that completes the task. Add no unrequested features.
4. Run `python -m pytest -q` after each change. Rerun `python scripts/build_all.py` when metrics or exports change.
5. Summarize what changed, what was checked, and anything that failed or looks off.
6. Stop and ask when a decision affects methodology (thresholds, smoothing, role definitions) rather than guessing.

## Commands

```powershell
pip install -r requirements.txt
python scripts/download_data.py          # all 10 games, about 380 MB, skips existing files
python scripts/download_data.py 114243   # one game (the test reference game)
python scripts/build_all.py              # about 70 s: results, WAIMS exports, figures
python -m pytest -q                      # 8 tests; the data tests skip if data/raw is missing
```

## Source data

* Upstream repo: https://github.com/SkillCorner/opendata-basketball (MIT license).
* Reference docs: copies of the upstream `PRIMER.md`, `data_dictionary/*.md` and `KNOWN_ISSUES.md` are in `reference/skillcorner_docs/`. When a field name or meaning is in doubt, check those files; never guess.
* Tracking files are Git LFS objects upstream. `scripts/download_data.py` fetches them over HTTPS from `media.githubusercontent.com`, so Git LFS is not needed.

## Hard rules

* **Never commit raw data.** `data/raw/` and `data/interim/` stay gitignored. Only the small CSVs and PNGs in `exports/` are committed.
* **Credit SkillCorner** in the README and on any public page.
* **Units stay in feet and ft/s internally.** Convert to meters only in presentation, and label every chart axis with units.
* **Anonymize players in WAIMS exports** ("Player NN"). Full names stay in `data/interim/results/`.
* **Keep WAIMS separate.** No WAIMS demo data in this repo, and no code here that WAIMS imports. WAIMS reads CSVs only.
* **No em dashes or en dashes** in prose, comments, docstrings or chart labels. Use commas, colons, parentheses or new sentences.
* **Charts:** role colors are guard `#2a78d6`, wing `#eb6834`, big `#1baf7a`. Text uses ink colors, never series colors. No dual axes.

## Data gotchas (verified)

* About half of all frames are dead time, with empty player lists. `io.parse_tracking` drops them.
* `gameClockStopped` separates running-clock frames from stopped-clock frames. Metrics use running clock; `minutes_all` also counts stopped-clock time on court.
* The provided `speed` matches speed derived from position (r = 0.998) but has rare spikes, up to about 59 ft/s. Capping at 26 ft/s affects about 50 frames per game.
* About 83% of positions are detected. Keep `detected` and `pred_error` on every frame, and report the detected share next to every metric.
* Event `frame` / `startFrame` / `endFrame` equal the tracking `frameIdx`.
* Player position is **not** in the data; roles are derived in `roles.py`.
* Pandas 3 stores string ids as a string dtype, not object. Map player id columns by explicit name.
* `closeouts.touchWallClock` is a string. `io.load_events` casts it.
* Some players have two `playerId` values across games. `io.load_aliases()` merges them, and it is applied in `clean.clean` and `actions.action_rows`.
* `passes.toReceiverId` holds the intercepting defender, not the receiver.

## Environment

* Windows with PowerShell locally, GitHub Desktop for commits.
* Division of labor: Claude handles strategy and product decisions; Claude Code handles implementation.
