# BUILD_SPEC: waims-tracking-lab

**Status:** v1 built and run on all 10 games (2026-09-28). The findings are in `README.md`.

## 1. The questions

1. What are the physical demands of a pro game, by role and by quarter?
2. What does each tactical action cost physically, for the player running it and for their defender?
3. What fills a player's most intense stretch?

## 2. Design decisions (and why)

| Decision | Choice | Reason |
|---|---|---|
| Time base | Running clock (game clock live) | Matches "live time" in basketball load research. Stopped-clock minutes are kept as `minutes_all`. |
| Speed source | Provided `speed`, capped at 26 ft/s | Matches position-derived speed (r = 0.998). Position-derived speed jumps to about 98 ft/s at extrapolation handoffs. |
| Smoothing | Savitzky Golay, 13 frames (0.52 s), order 2, within segments | Keeps real accelerations. Sensitivity to the filter is published in the README. |
| Speed zones (ft/s) | Z1 under 7, Z2 7 to 12, Z3 12 to 16, Z4 16 to 20, Z5 20 and above | The 90th and 99th speed percentiles are 12.8 and 18.6 ft/s. HSD = Z4 + Z5. |
| Effort threshold | 9.8 ft/s² (about 3 m/s²) for at least 0.3 s | A common threshold in indoor court-sport research. |
| Sprint | 20+ ft/s for at least 0.5 s | Z5 entry. The median player-game has 1 to 2 sprints. |
| Peak windows | Rolling 30 s and 60 s of running clock, within one period, game-clock span at most window + 2 s | Stops windows from spanning a stint on the bench. |
| Roles | Scoring rule on per-36 event features (default), with k-means (k = 3) as a check | Positions are not published. The rule is explainable, and k-means agrees on 80% of players. |
| Minimum minutes | 10 total running minutes for a role, 10 per game for benchmarks | Keeps garbage-time noise out of the bands. |
| Action window | Span actions use their own frames; point actions use frame ± 2 s | Picks, handoffs and off-ball screens are single-frame events. |
| Action cost | Excess over the player's own running-clock mean speed in the same game | Controls for players who are always fast. |

## 3. Results table map

Everything is written by `scripts/build_all.py`.

| File | Grain |
|---|---|
| `data/interim/results/player_game.csv` | player-game, all metrics, real names |
| `data/interim/results/player_period.csv` | player-period |
| `data/interim/results/peak_windows.csv` | top 3 windows per player-game, 30 s and 60 s |
| `data/interim/results/action_windows.csv` | one row per action x involved player |
| `data/interim/results/roles.csv`, `role_summary.csv` | player features, role labels, role means |
| `data/interim/results/peak_window_context.csv` | actions per peak minute vs per average minute |
| `data/interim/results/smoothing_sensitivity.csv` | player-game metrics under 5 filters |
| `exports/waims/*.csv` | WAIMS contract (section 4) |
| `exports/figures/*.png` | 6 figures used in the README |

## 4. WAIMS export contract

WAIMS reads only these files. Units are feet. Players are anonymized.

* **`game_demands_player.csv`** (player-game): `game_id, game_date, team_name, player_label, role, minutes_running, distance_ft, distance_per_min, hsd_ft, hsd_per_min, sprints, accels, decels, efforts_per_min, peak_speed, peak30_dist_per_min, peak60_dist_per_min, detected_share`. `role` is `guard`, `wing`, `big` or `unassigned` (under 10 total minutes).
* **`role_benchmarks.csv`**: `role, metric, p10, p50, p90, n_player_games`.
* **`role_by_quarter.csv`**: `role, period, distance_per_min, efforts_per_min`. Period 5 is overtime.
* **`action_cost.csv`**: `action, actor, player_role, n, mean_duration_s, mean_speed, mean_excess_speed, mean_excess_distance_ft, efforts_per_min, mean_peak_speed`. Rows with n under 10 are dropped.

**WAIMS view (separate task, in the WAIMS repo):** add a "Game Demands (real pro data)" tab with:
* role benchmark bands (p10 to p90, with a p50 marker) for distance per min, HSD per min, efforts per min and peak 60 s
* a quarter profile by role
* an action cost table

Label the tab as SkillCorner ACB data, kept apart from the synthetic demo roster. Use the WAIMS UI conventions: 🟢🟡🔴, text status labels, horizontal fill bars.

## 5. Backlog (in priority order)

1. **WAIMS tab** (in the WAIMS repo), per section 4.
2. **Portfolio case study page** (in the portfolio repo): problem, method, 3 figures (demands by role, action cost, peak context), "how a staff would use this", link to the GitHub repo.
3. **Worst-case scenario curve:** peak distance per minute across window lengths of 15, 30, 60, 120 and 180 s, by role. It is a standard load-monitoring figure and a natural extension of `peak_windows`.
4. **Confidence intervals** on the action cost table (bootstrap by game), so small cells are flagged.
5. **Score and time context:** load in close games vs blowouts, and in the last 5 minutes. Use `possessions.homeStartScore` / `awayStartScore`.
6. **Aggregates side project:** the 293-game shot, drive and pick aggregates could feed a WAIMS-GM scouting demo. Keep it out of this repo unless it uses the tracking data.

## 6. Limitations to keep stated

About 17% of positions are extrapolated; players are held at z = 0 (no jump or contact load); effort counts depend on the filter (up to 14%); the sample is 10 games of ACB men's pro basketball, and transfer to other leagues is an assumption.
