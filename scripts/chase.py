"""Chase Cost vs gravity (pick 3).

Usage:  python scripts/chase.py
Outputs:
  data/interim/results/chase_player_game.csv   per offensive player-game, with names (gitignored)
  data/interim/results/chase_player.csv        pooled per player, with names (gitignored)
  exports/chase/chase_player.csv               pooled per player, anonymized ("Player NN")
  exports/chase/chase_checks.csv               reliability and the gravity model fit
  exports/chase/chase_vs_gravity.png
  exports/chase/chase_by_role.csv, chase_by_role.png   by role of the player guarded, on and off the ball

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

from wtl import chase, clean, closeouts, io  # noqa: E402
from wtl.viz import INK, INK2, ROLE_COLOR, ROLE_ORDER, SURFACE, _finish, plt  # noqa: E402

RESULTS = ROOT / "data" / "interim" / "results"
OUT = ROOT / "exports" / "chase"


def figure(p: pd.DataFrame, out: Path) -> None:
    d = p[p["role"].isin(ROLE_ORDER)]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for role in ROLE_ORDER:
        s = d[d["role"] == role]
        ax.scatter(s["gravity_ft"], s["chase_cost_ft_per_min"], s=14 + s["minutes_guarded"] * 0.8,
                   color=ROLE_COLOR[role], alpha=0.7, edgecolors=SURFACE, linewidths=0.6, label=role)
    ax.axvline(d["gravity_ft"].median(), color=INK2, lw=1, ls="--")
    ax.axhline(d["chase_cost_ft_per_min"].median(), color=INK2, lw=1, ls="--")
    for (xa, ya, txt) in ((0.98, 0.97, "High gravity, high chase cost"), (0.02, 0.97, "Low gravity, high chase cost"),
                          (0.98, 0.03, "High gravity, low chase cost"), (0.02, 0.03, "Low gravity, low chase cost")):
        ax.text(xa, ya, txt, transform=ax.transAxes, ha="right" if xa > 0.5 else "left",
                va="top" if ya > 0.5 else "bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel("Gravity: ft tighter than court position predicts (off the ball)")
    ax.set_ylabel("Chase Cost: defender ft/min above their own average")
    ax.legend(frameon=False, loc="center right", labelcolor=INK2)
    _finish(fig, out / "chase_vs_gravity.png", "Chase Cost vs gravity: two different ways to tax a defense",
            f"One dot per player with {chase.MIN_GUARDED_MIN:.0f}+ min guarded, pooled over games; size = minutes. "
            "Dashed lines = medians.")


def role_figure(t: pd.DataFrame, out: Path) -> None:
    panels = [("off", "Off the ball"), ("on", "With the ball")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(ROLE_ORDER))[::-1]
    for ax, (st, title) in zip(axes, panels):
        s = t[t["ball"] == st].set_index("role").reindex(ROLE_ORDER)
        for yi, (role, r) in zip(y, s.iterrows()):
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=ROLE_COLOR[role], lw=1.5, alpha=0.6)
            ax.plot(r["chase_cost_ft_per_min"], yi, "o", color=ROLE_COLOR[role], ms=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{r['chase_cost_ft_per_min']:+.0f}", (r["chase_cost_ft_per_min"], yi), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{title}  ({s['minutes'].sum():.0f} min guarded)", loc="left", fontsize=10)
        ax.set_xlabel("Defender ft/min above their own average")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(ROLE_ORDER) - 0.3)
    axes[0].set_yticks(y, [f"Guarding a {r}" for r in ROLE_ORDER])
    _finish(fig, out / "chase_by_role.png", "Who you guard sets how much you run",
            "Matchup defenders vs their own average in the same state (on or off the ball). "
            "Lines = 95% interval, bootstrap by game. Roles derived from on-court actions.")


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    aliases = io.load_aliases()
    games = io.load_matches()["game_id"].tolist()

    braking = {}
    for gid in games:
        for pid, arr in closeouts.decel_ceiling(clean.clean(io.tracking(gid), aliases)).items():
            braking.setdefault(pid, []).append(arr)
    ceiling = {pid: np.quantile(np.concatenate(a), closeouts.DECEL_CEIL_Q) for pid, a in braking.items()}

    rows, samples = [], []
    for gid in games:
        r, s = chase.game(clean.clean(io.tracking(gid), aliases), io.load_events(gid), chase.ball(gid),
                          ceiling, aliases)
        rows.append(r)
        samples.append(s)
        print(f"game {gid}: {len(r)} matchup intervals ({time.time() - t0:.0f}s)", flush=True)
    cc = chase.chase_cost(pd.concat(rows, ignore_index=True))
    gv, r2 = chase.gravity(pd.concat(samples, ignore_index=True))
    pg = cc.merge(gv, on=["game_id", "off_id"], how="left").rename(columns={"off_id": "player_id"})
    pg = pg[pg["minutes_guarded"] >= chase.MIN_GUARDED_MIN]

    roster = pd.concat([io.load_roster(g) for g in games])
    roster["player_id"] = roster["player_id"].map(lambda p: aliases.get(p, p))
    names = roster.drop_duplicates("player_id").set_index("player_id")[["player_name", "team_name"]]
    roles = pd.read_csv(RESULTS / "roles.csv", index_col=0)["role"]
    pg = pg.join(names, on="player_id")
    pg["role"] = pg["player_id"].map(roles)
    pg.to_csv(RESULTS / "chase_player_game.csv", index=False)

    # Pool player-games, each metric weighted by the minutes it is measured over
    weights = {"chase_cost_ft_per_min": "minutes_guarded", "chase_cost_decels_per_min": "minutes_guarded",
               "gravity_ft": "minutes_guarded",
               "chase_cost_on_ball_ft_per_min": "minutes_on_ball", "chase_cost_on_ball_decels_per_min": "minutes_on_ball",
               "chase_cost_off_ball_ft_per_min": "minutes_off_ball",
               "chase_cost_off_ball_decels_per_min": "minutes_off_ball"}
    grp = pg.groupby("player_id")
    p = grp.agg(games=("game_id", "size"), minutes_guarded=("minutes_guarded", "sum"),
                minutes_on_ball=("minutes_on_ball", "sum"), minutes_off_ball=("minutes_off_ball", "sum"))
    for c, wcol in weights.items():
        ok = pg[c].notna()
        p[c] = (pg[c] * pg[wcol]).where(ok).groupby(pg["player_id"]).sum() \
            / pg[wcol].where(ok).groupby(pg["player_id"]).sum()
    p = p.join(names)
    p["role"] = p.index.map(roles)
    p.to_csv(RESULTS / "chase_player.csv")
    labels = {pid: f"Player {i + 1:02d}" for i, pid in enumerate(sorted(p.index))}
    p.drop(columns=["player_name", "team_name"]).rename(index=labels).rename_axis("player_label") \
        .round(3).to_csv(OUT / "chase_player.csv")

    checks = [chase.reliability(pg, c) for c in weights]
    q = p.dropna(subset=["gravity_ft"])
    checks.append({"metric": "chase_vs_gravity_spearman", "players": len(q),
                   "spearman": q["chase_cost_ft_per_min"].corr(q["gravity_ft"], method="spearman")})
    checks.append({"metric": "off_ball_chase_vs_gravity_spearman", "players": len(q),
                   "spearman": q["chase_cost_off_ball_ft_per_min"].corr(q["gravity_ft"], method="spearman")})
    checks.append({"metric": "on_vs_off_ball_chase_spearman", "players": len(p),
                   "spearman": p["chase_cost_on_ball_ft_per_min"].corr(p["chase_cost_off_ball_ft_per_min"],
                                                                      method="spearman")})
    checks.append({"metric": "gravity_model_r2", "pearson": r2})
    pd.DataFrame(checks).round(3).to_csv(OUT / "chase_checks.csv", index=False)
    figure(p, OUT)
    rt = chase.role_table(pg)
    rt.round(3).to_csv(OUT / "chase_by_role.csv", index=False)
    role_figure(rt, OUT)
    print(pd.DataFrame(checks).round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
