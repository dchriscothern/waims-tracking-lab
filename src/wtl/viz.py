"""Static figures for the README and portfolio (matplotlib, light surface).

Palette follows the validated reference instance: role hues use the first three
categorical slots (validated for all-pairs use), diverging is blue <-> red with a
neutral gray midpoint. Text uses ink tokens, never series colors.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
ROLE_ORDER = ["guard", "wing", "big"]
ROLE_COLOR = {"guard": "#2a78d6", "wing": "#eb6834", "big": "#1baf7a"}
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SOURCE = "Data: SkillCorner Open Data, Liga ACB 2025/26 (10 games). Running-clock time only."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def _finish(fig, path: Path, title: str, subtitle: str | None = None) -> None:
    h = fig.get_size_inches()[1]
    head = 0.95 / h                      # inches reserved for title block, as a fraction
    fig.tight_layout(rect=(0, 0.35 / h, 1, 1 - head))
    fig.text(0.01, 1 - 0.3 / h, title, ha="left", va="center", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.62 / h, subtitle, ha="left", va="center", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.1 / h, SOURCE, ha="left", va="center", fontsize=8, color=INK2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def demands_by_role(pg: pd.DataFrame, peaks: pd.DataFrame, out: Path) -> None:
    peak60 = peaks[(peaks["window_s"] == 60) & (peaks["rank"] == 1)][["game_id", "player_id", "distance_per_min"]] \
        .rename(columns={"distance_per_min": "peak60"})
    d = pg.merge(peak60, on=["game_id", "player_id"], how="left")
    d = d[(d["minutes_running"] >= 10) & d["role"].isin(ROLE_ORDER)]
    panels = [("distance_per_min", "Distance (ft/min)"), ("hsd_per_min", "High-speed distance\n16+ ft/s (ft/min)"),
              ("efforts_per_min", "Accels + decels\nabove 9.8 ft/s² (per min)"), ("peak60", "Peak 60 s distance\n(ft/min)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    rng = np.random.default_rng(3)
    for ax, (col, label) in zip(axes, panels):
        for i, role in enumerate(ROLE_ORDER):
            vals = d.loc[d["role"] == role, col].dropna()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=ROLE_COLOR[role],
                       alpha=0.55, edgecolors=SURFACE, linewidths=0.5)
            med = vals.median()
            ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2, solid_capstyle="round")
            ax.annotate(f"{med:.0f}" if med >= 10 else f"{med:.2f}", (i + 0.32, med), va="center",
                        fontsize=8.5, color=INK)
        ax.set_xticks(range(3), [f"{r}\nn={int((d['role'] == r).sum())}" for r in ROLE_ORDER])
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.grid(axis="x", visible=False)
    _finish(fig, out / "demands_by_role.png", "Game physical demands by role",
            "Each dot is one player-game (10+ running-clock minutes). Black line = median.")


def quarter_profile(per: pd.DataFrame, out: Path) -> None:
    q = per[(per["minutes"] >= 3) & (per["period"] <= 4) & per["role"].isin(ROLE_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, label) in zip(axes, [("distance_per_min", "Distance (ft/min)"),
                                       ("efforts_per_min", "Accels + decels (per min)")]):
        m = q.groupby(["role", "period"])[col].median().unstack(0)
        for role in ROLE_ORDER:
            ax.plot(m.index, m[role], color=ROLE_COLOR[role], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        # End labels, nudged apart so they never overlap
        ends = sorted(((m[r].iloc[-1], r) for r in ROLE_ORDER))
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.07
        placed = []
        for val, role in ends:
            y = max(val, placed[-1] + gap) if placed else val
            placed.append(y)
            ax.annotate(role, (m.index[-1] + 0.08, y), va="center", fontsize=9, color=INK2)
        ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlim(0.8, 4.5)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    _finish(fig, out / "quarter_profile.png", "Intensity by quarter",
            "Median per player-quarter (3+ running minutes).")


def action_cost_heatmap(cost: pd.DataFrame, out: Path) -> None:
    c = cost.copy()
    w = c.assign(wx=c["mean_excess_speed"] * c["n"]).groupby(["action", "actor"]) \
        .agg(wx=("wx", "sum"), n=("n", "sum"))
    w["excess"] = w["wx"] / w["n"]
    w = w.reset_index()
    w["side"] = np.where(w["actor"].str.endswith("_def"), "Defender", "Offense")
    w["who"] = w["actor"].str.replace("_def", "", regex=False)
    order = ["drives", "closeouts", "isolations", "posts", "picks", "handoffs", "off_ball_screens"]
    rows = [a for a in order if a in set(w["action"])]
    labels, vals, ns = [], [], []
    for a in rows:
        for who in w.loc[w["action"] == a, "who"].unique():
            labels.append(f"{a.replace('_', ' ')}: {who}")
            vals.append([w.loc[(w["action"] == a) & (w["who"] == who) & (w["side"] == s), "excess"].squeeze()
                         if ((w["action"] == a) & (w["who"] == who) & (w["side"] == s)).any() else np.nan
                         for s in ("Offense", "Defender")])
            ns.append([int(w.loc[(w["action"] == a) & (w["who"] == who) & (w["side"] == s), "n"].sum())
                       for s in ("Offense", "Defender")])
    arr = np.array(vals, dtype=float)
    lim = np.nanmax(np.abs(arr))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.6))
    ax.imshow(arr, cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(2):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:+.1f}  (n={ns[i][j]})", ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xticks([0, 1], ["Player with the action", "Their defender"])
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _finish(fig, out / "action_cost.png", "What an action costs: speed above the player's own average",
            "Mean speed in the action window minus that player's running-clock mean (ft/s). Red = harder.")


def pick_coverage(meas: pd.DataFrame, out: Path) -> None:
    p = meas[(meas["action"] == "picks") & meas["actor"].isin(["ballhandler_def", "screener_def"])].dropna(
        subset=["coverage"])
    g = p.groupby(["actor", "coverage"]).agg(n=("action_id", "size"), ex=("excess_speed", "mean"),
                                              se=("excess_speed", lambda s: s.std() / np.sqrt(len(s))))
    g = g[g["n"] >= 20].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, actor, title in zip(axes, ["ballhandler_def", "screener_def"],
                                ["Ball handler's defender", "Screener's defender"]):
        s = g[g["actor"] == actor].sort_values("ex")
        y = np.arange(len(s))
        ax.errorbar(s["ex"], y, xerr=1.96 * s["se"], fmt="o", color="#2a78d6", ms=7, ecolor=INK2,
                    elinewidth=1, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(y, [f"{c} (n={n})" for c, n in zip(s["coverage"], s["n"])])
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Excess speed in the pick window (ft/s), 95% CI")
        ax.grid(axis="y", visible=False)
    _finish(fig, out / "pick_coverage.png", "Pick coverage and defender workload",
            "Coverage types with 20+ picks across the 10 games.")


def peak_context(ctx: pd.DataFrame, out: Path) -> None:
    c = ctx[ctx["role"].isin(ROLE_ORDER)].copy()
    c["label"] = c["action"].str.replace("_", " ") + np.where(c["actor_side"] == "def", " (defending)", "")
    keep = c.groupby("label")["per_window"].max().nlargest(9).index
    c = c[c["label"].isin(keep)]
    order = c.groupby("label")["per_window"].sum().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(order))
    for k, role in enumerate(ROLE_ORDER):
        s = c[c["role"] == role].set_index("label").reindex(order)
        ax.barh(y + (1 - k) * 0.27, s["per_window"], height=0.25, color=ROLE_COLOR[role], label=role,
                edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, order)
    ax.set_xlabel("Involvements per player's most intense 60 s")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    ax.grid(axis="y", visible=False)
    _finish(fig, out / "peak_context.png", "What fills a player's most intense minute",
            "Top 60 s running-clock window per player-game, actions the player took part in.")


def smoothing(sens: pd.DataFrame, out: Path) -> None:
    s = sens[sens["minutes_running"] >= 10].groupby("method")[
        ["distance_per_min", "accels_per_min", "decels_per_min"]].median()
    order = ["none", "savgol_7", "savgol_13", "savgol_25", "butter"]
    s = s.reindex([o for o in order if o in s.index])
    names = {"none": "No smoothing", "savgol_7": "Savitzky Golay 0.28 s", "savgol_13": "Savitzky Golay 0.52 s (used)",
             "savgol_25": "Savitzky Golay 1.0 s", "butter": "Butterworth 1 Hz"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    y = np.arange(len(s))
    for ax, col, label in zip(axes, s.columns, ["Distance (ft/min)", "Accels (per min)", "Decels (per min)"]):
        colors = ["#2a78d6" if i == "savgol_13" else "#9ec5f4" for i in s.index]
        ax.barh(y, s[col], color=colors, height=0.6)
        for yi, v in zip(y, s[col]):
            ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.2f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [names[i] for i in s.index])
    _finish(fig, out / "smoothing_sensitivity.png",
            "Filter choice leaves distance unchanged but moves effort counts by up to 14%",
            "Median per player-game. Report effort counts together with the filter used.")


def all_figures(pg, per, peaks, meas, sens, cost, ctx, rl, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    demands_by_role(pg, peaks, out)
    quarter_profile(per, out)
    action_cost_heatmap(cost, out)
    pick_coverage(meas, out)
    peak_context(ctx, out)
    smoothing(sens, out)
