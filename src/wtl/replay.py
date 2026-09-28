"""Animated 2D replay of a player's window (e.g. their peak minute) as an MP4.

Left: FIBA court with all ten players and the ball; the focus player carries a
fading speed trail. Filled dots are detected positions, hollow dots are
extrapolated. Right: live speed, distance, pace against the game average,
effort count, a speed trace and a log of the actions the player takes part in.

Data: SkillCorner Open Data (MIT). Rendering: matplotlib + ffmpeg.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle  # noqa: E402

from .io import FPS, match_dir  # noqa: E402

# Dark video palette (dark-mode steps of the reference palette)
BG = "#1a1a19"
FLOOR = "#222220"
LINE = "#55544f"
INK = "#ffffff"
INK2 = "#c3c2b7"
INK3 = "#8d8c84"
TEAM = "#3987e5"          # focus player's teammates
OPP = "#d95926"           # opponents
FOCUS = "#199e70"         # focus player
BALL = "#f2c14e"
ZONE_FILL = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]   # sequential, Z1..Z5
ZONE_EDGES = [0, 7, 12, 16, 20, 26]

L, W = 45.93, 24.61                   # half length / half width, ft (28 m x 15 m)
HOOP = 40.76
TRAIL_S = 3.0

ACTION_LABEL = {
    ("drives", "ballhandler"): "Drive",
    ("drives", "ballhandler_def"): "Guarding a drive",
    ("isolations", "ballhandler"): "Isolation",
    ("isolations", "ballhandler_def"): "Guarding an isolation",
    ("posts", "ballhandler"): "Post-up",
    ("posts", "ballhandler_def"): "Guarding a post-up",
    ("closeouts", "ballhandler"): "Attacking a closeout",
    ("closeouts", "ballhandler_def"): "Closeout",
    ("picks", "ballhandler"): "Pick and roll: ball handler",
    ("picks", "screener"): "Pick and roll: screener",
    ("picks", "ballhandler_def"): "Guarding the ball in a pick",
    ("picks", "screener_def"): "Guarding the screener",
    ("handoffs", "setter"): "Handoff: giving",
    ("handoffs", "receiver"): "Handoff: receiving",
    ("handoffs", "setter_def"): "Guarding a handoff",
    ("handoffs", "receiver_def"): "Guarding a handoff",
    ("off_ball_screens", "cutter"): "Cutting off a screen",
    ("off_ball_screens", "screener"): "Setting an off-ball screen",
    ("off_ball_screens", "cutter_def"): "Chasing through a screen",
    ("off_ball_screens", "screener_def"): "Guarding the screener",
}
POINT_ACTIONS = {"picks", "handoffs", "off_ball_screens"}


@dataclass
class Clip:
    game_id: int
    player_id: int
    start_frame: int
    end_frame: int
    title: str
    subtitle: str
    role: str
    period: int
    game_avg_pace: float          # ft per running minute over the whole game
    game_avg_efforts: float       # efforts per running minute over the whole game


def draw_court(ax) -> None:
    ax.set_facecolor(BG)
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, facecolor=FLOOR, edgecolor=LINE, lw=2, zorder=0))
    kw = dict(color=LINE, lw=1.6, zorder=1)
    ax.plot([0, 0], [-W, W], **kw)
    ax.add_patch(Circle((0, 0), 5.91, fill=False, edgecolor=LINE, lw=1.6, zorder=1))
    for s in (-1, 1):
        hx = s * HOOP
        # key and free-throw circle
        kx = s * (L - 19.03)
        ax.add_patch(Rectangle((min(kx, s * L), -8.04), 19.03, 16.08, fill=False, edgecolor=LINE, lw=1.6, zorder=1))
        ax.add_patch(Circle((kx, 0), 5.91, fill=False, edgecolor=LINE, lw=1.6, zorder=1))
        # three-point line: corner segments plus arc
        cx = s * (HOOP - 4.68)
        ax.plot([s * L, cx], [21.65, 21.65], **kw)
        ax.plot([s * L, cx], [-21.65, -21.65], **kw)
        ang = np.degrees(np.arctan2(21.65, 4.68))
        t1, t2 = (180 - ang, 180 + ang) if s > 0 else (-ang, ang)
        ax.add_patch(Arc((hx, 0), 44.3, 44.3, theta1=t1, theta2=t2, edgecolor=LINE, lw=1.6, zorder=1))
        # restricted area, backboard, rim
        r1, r2 = (90, 270) if s > 0 else (-90, 90)
        ax.add_patch(Arc((hx, 0), 8.2, 8.2, theta1=r1, theta2=r2, edgecolor=LINE, lw=1.6, zorder=1))
        ax.plot([s * 41.99, s * 41.99], [-3, 3], color=INK3, lw=2.5, zorder=1)
        ax.add_patch(Circle((hx, 0), 0.75, fill=False, edgecolor=INK3, lw=2, zorder=1))
    ax.set_xlim(-L - 1.5, L + 1.5)
    ax.set_ylim(-W - 1.5, W + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def load_ball(game_id: int, start: int, end: int) -> pd.DataFrame:
    rows = []
    with gzip.open(match_dir(game_id) / f"{game_id}_tracking_data.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if '"frameIdx"' not in line:
                continue
            fr = json.loads(line)
            f = fr["frameIdx"]
            if f < start:
                continue
            if f > end:
                break
            b = fr.get("ball") or {}
            if b.get("xyz"):
                rows.append((f, b["xyz"][0], b["xyz"][1], b["xyz"][2]))
    return pd.DataFrame(rows, columns=["frame", "x", "y", "z"]).set_index("frame")


def action_spans(acts: pd.DataFrame) -> pd.DataFrame:
    """Display spans and labels for the focus player's actions."""
    a = acts.copy()
    point = a["action"].isin(POINT_ACTIONS)
    # point events were stored as frame +/- 2 s; show from 1 s before to 2 s after the event frame
    a["show_start"] = np.where(point, a["start_frame"] + FPS, a["start_frame"])
    a["show_end"] = np.where(point, a["end_frame"], a["end_frame"])
    a["label"] = [ACTION_LABEL.get((r.action, r.actor), r.action) for r in a.itertuples()]
    cov = a["coverage"].where(a["coverage"].notna() & (a["coverage"].astype(str) != "nan"), "")
    a["short"] = a["label"]
    a["label"] = np.where(cov != "", a["label"] + " (" + cov.astype(str) + ")", a["label"])
    a["defense"] = a["actor"].str.endswith("_def")
    return a.sort_values("show_start").reset_index(drop=True)


def _pill(fig, x, y, text, color):
    return fig.text(x, y, text, fontsize=15, color=INK, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", fc=color, ec="none"))


class _Stills:
    """Stand-in for FFMpegWriter that saves chosen frames as PNGs (for checking layout)."""

    def __init__(self, fig, out: Path, keep: set[int]):
        self.fig, self.out, self.keep, self.n = fig, out, keep, 0

    def saving(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()

    def grab_frame(self):
        if self.n in self.keep:
            self.fig.savefig(self.out.with_name(f"{self.out.stem}_f{self.n:05d}.png"), facecolor=BG)
        self.n += 1


def render(clip: Clip, players: pd.DataFrame, focus: pd.DataFrame, acts: pd.DataFrame,
           out: Path, intro_s: float = 3.0, outro_s: float = 5.0, dpi: int = 100,
           stills: list[int] | None = None) -> Path:
    """players: raw player-frame rows for the window (all 10 players).
    focus: cleaned + flagged rows for the focus player over the window (v, a, efforts).
    acts: measured action rows for the focus player overlapping the window.
    """
    frames = np.arange(clip.start_frame, clip.end_frame + 1)
    by_frame = {f: g for f, g in players.groupby("frame")}
    ball = load_ball(clip.game_id, clip.start_frame, clip.end_frame)
    focus = focus.set_index("frame").reindex(frames)
    v = focus["v"].to_numpy()
    eff_start = (focus["accel_start"].fillna(False) | focus["decel_start"].fillna(False)).to_numpy()
    acc_start = focus["accel_start"].fillna(False).to_numpy()
    dist_cum = np.nancumsum(v / FPS)
    eff_cum = np.cumsum(eff_start)
    t = (frames - frames[0]) / FPS
    focus_side = players.loc[players["player_id"] == clip.player_id, "side"].iloc[0]
    spans = action_spans(acts)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor=BG)
    if stills is not None:
        writer = _Stills(fig, out, set(stills))
    else:
        writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=9000,
                              extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-movflags", "+faststart"])

    # ---------- layout ----------
    court = fig.add_axes([0.02, 0.24, 0.63, 0.62])
    draw_court(court)
    fig.text(0.02, 0.945, clip.title, fontsize=30, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.900, clip.subtitle, fontsize=16, color=INK2, va="center")
    fig.text(0.02, 0.025, "Data: SkillCorner Open Data, Liga ACB 2025/26. Broadcast tracking, 25 fps. "
             "Hollow dots = position estimated off-camera.", fontsize=11, color=INK3, va="center")
    fig.text(0.98, 0.025, "Chris Cothern  |  waims-tracking-lab", fontsize=11, color=INK3, va="center", ha="right")
    # legend
    lx, ly = 0.02, 0.215
    for i, (lab, col) in enumerate([("Focus player", FOCUS), ("Teammates", TEAM), ("Opponents", OPP), ("Ball", BALL)]):
        fig.add_artist(Line2D([lx + i * 0.095 + 0.006], [ly], marker="o", ms=11, color=col, lw=0,
                              transform=fig.transFigure))
        fig.text(lx + i * 0.095 + 0.016, ly, lab, fontsize=13, color=INK2, va="center")
    fig.text(0.02, 0.165, "NOW", fontsize=12, color=INK3, va="center", fontweight="bold")

    # right panel
    px = 0.685
    fig.text(px, 0.945, "ELAPSED", fontsize=12, color=INK3, fontweight="bold", va="center")
    t_elapsed = fig.text(px, 0.900, "", fontsize=26, color=INK, fontweight="bold", va="center")
    fig.text(px + 0.14, 0.945, "GAME CLOCK", fontsize=12, color=INK3, fontweight="bold", va="center")
    t_clock = fig.text(px + 0.14, 0.900, "", fontsize=26, color=INK, fontweight="bold", va="center")

    fig.text(px, 0.835, "SPEED", fontsize=12, color=INK3, fontweight="bold", va="center")
    t_speed = fig.text(px, 0.785, "", fontsize=40, color=INK, fontweight="bold", va="center")
    t_speed2 = fig.text(px + 0.295, 0.835, "", fontsize=15, color=INK2, va="center", ha="right")
    bar = fig.add_axes([px, 0.715, 0.295, 0.028])
    bar.set_xlim(0, 26)
    bar.set_ylim(0, 1)
    bar.axis("off")
    bar.add_patch(FancyBboxPatch((0, 0), 26, 1, boxstyle="round,pad=0,rounding_size=0.5", fc="#2e2e2c", ec="none"))
    fill = FancyBboxPatch((0, 0), 0.01, 1, boxstyle="round,pad=0,rounding_size=0.5", fc=ZONE_FILL[0], ec="none")
    bar.add_patch(fill)
    for edge in (16, 20):
        bar.plot([edge, edge], [-0.3, 1.3], color=INK3, lw=1, clip_on=False)
    fig.text(px + 0.295 * 16 / 26, 0.695, "16 high speed", fontsize=10, color=INK3, ha="center", va="center")
    fig.text(px + 0.295 * 20 / 26, 0.675, "20 sprint", fontsize=10, color=INK3, ha="center", va="center")

    tiles = [("DISTANCE", "ft this minute"), ("PACE", "ft/min so far"), ("EFFORTS", "hard accels + decels")]
    tval = []
    for i, (h, sub) in enumerate(tiles):
        x0 = px + i * 0.103
        fig.text(x0, 0.625, h, fontsize=12, color=INK3, fontweight="bold", va="center")
        tval.append(fig.text(x0, 0.580, "", fontsize=30, color=INK, fontweight="bold", va="center"))
        fig.text(x0, 0.540, sub, fontsize=11, color=INK3, va="center")
    t_vsavg = fig.text(px + 0.103, 0.510, "", fontsize=12, color=INK2, va="center")
    t_effavg = fig.text(px + 0.206, 0.510, f"avg {clip.game_avg_efforts:.1f}/min", fontsize=12, color=INK2,
                        va="center")

    tr = fig.add_axes([px + 0.02, 0.265, 0.275, 0.2], facecolor=BG)
    tr.set_xlim(0, t[-1])
    tr.set_ylim(0, 22)
    for s in ("top", "right"):
        tr.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        tr.spines[s].set_color(LINE)
    tr.tick_params(colors=INK3, labelsize=11)
    tr.set_xlabel("seconds", color=INK3, fontsize=11)
    tr.set_ylabel("ft/s", color=INK3, fontsize=11)
    tr.axhline(16, color=LINE, lw=1, ls=(0, (4, 4)))
    tr.plot(t, v, color="#383835", lw=1.5)
    (trace,) = tr.plot([], [], color=FOCUS, lw=2.2)
    eff_dots = tr.scatter([], [], s=34, color=INK, zorder=3, edgecolors=BG, linewidths=1)
    cursor = tr.axvline(0, color=INK2, lw=1)
    fig.text(px, 0.485, "SPEED TRACE", fontsize=12, color=INK3, fontweight="bold", va="center")
    fig.text(px + 0.295, 0.485, "white dot = hard accel / decel", fontsize=10, color=INK3, va="center", ha="right")

    fig.text(px, 0.195, "ACTIONS THIS MINUTE", fontsize=12, color=INK3, fontweight="bold", va="center")
    log_rows = [fig.text(px + (0.158 if i >= 5 else 0), 0.160 - (i % 5) * 0.026, "", fontsize=11.5, color=INK2,
                         va="center") for i in range(10)]

    # court artists
    trail = LineCollection([], linewidths=[], capstyle="round", zorder=3)
    court.add_collection(trail)
    dots = court.scatter([], [], s=[], zorder=4)
    ring = court.scatter([], [], s=900, facecolors="none", edgecolors=INK, linewidths=2.2, zorder=5)
    bdot = court.scatter([], [], s=90, color=BALL, zorder=6, edgecolors=BG, linewidths=1.2)
    pills: list = []

    # intro/outro cards live on their own figure; the writer is pointed at it while they play
    cfig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor=BG)
    c_texts = []

    def card(lines):
        for tt in c_texts:
            tt.remove()
        c_texts.clear()
        for (y, s, size, col, bold) in lines:
            c_texts.append(cfig.text(0.5, y, s, fontsize=size, color=col, ha="center", va="center",
                                     fontweight="bold" if bold else "normal"))
        writer.fig = cfig

    def dashboard():
        writer.fig = fig

    out.parent.mkdir(parents=True, exist_ok=True)
    trail_n = int(TRAIL_S * FPS)
    with writer.saving(fig, str(out), dpi):
        # ---------- intro ----------
        card([(0.62, clip.title, 46, INK, True),
              (0.54, clip.subtitle, 22, INK2, False),
              (0.42, "His most intense 60 seconds of running clock, replayed in real time.", 20, INK2, False),
              (0.37, "Tracking from broadcast video. Every number is computed from player positions.", 16, INK3, False),
              (0.10, "Data: SkillCorner Open Data (Liga ACB 2025/26)  |  Chris Cothern, waims-tracking-lab", 13, INK3,
               False)])
        for _ in range(int(intro_s * FPS)):
            writer.grab_frame()
        dashboard()

        # ---------- replay ----------
        for i, f in enumerate(frames):
            g = by_frame.get(f)
            if g is not None:
                xy = g[["x", "y"]].to_numpy()
                isf = (g["player_id"] == clip.player_id).to_numpy()
                team = (g["side"] == focus_side).to_numpy()
                base = np.where(isf, FOCUS, np.where(team, TEAM, OPP))
                det = g["detected"].to_numpy()
                face = np.where(det, base, FLOOR)
                dots.set_offsets(xy)
                dots.set_facecolors(face)
                dots.set_edgecolors(base)
                dots.set_linewidths(np.where(det, 0.0, 2.5))
                dots.set_sizes(np.where(isf, 420, 260))
                ring.set_offsets(xy[isf])
            # trail
            lo = max(0, i - trail_n)
            fx = focus["x"].to_numpy()[lo:i + 1]
            fy = focus["y"].to_numpy()[lo:i + 1]
            if len(fx) > 1:
                seg = np.stack([np.column_stack([fx[:-1], fy[:-1]]), np.column_stack([fx[1:], fy[1:]])], axis=1)
                age = np.linspace(0.08, 1, len(seg))
                trail.set_segments(seg)
                col = np.tile(matplotlib.colors.to_rgba(FOCUS), (len(seg), 1))
                col[:, 3] = age
                trail.set_color(col)
                trail.set_linewidths(1 + 5 * age)
            if f in ball.index:
                bdot.set_offsets([[ball.at[f, "x"], ball.at[f, "y"]]])
            # panel
            sp = v[i] if np.isfinite(v[i]) else 0.0
            zone = int(np.clip(np.searchsorted(ZONE_EDGES, sp, side="right") - 1, 0, 4))
            fill.set_width(max(sp, 0.01))
            fill.set_facecolor(ZONE_FILL[zone])
            t_elapsed.set_text(f"0:{int(t[i]):02d}")
            gc = focus["game_clock"].iloc[i] if "game_clock" in focus else np.nan
            if np.isfinite(gc):
                t_clock.set_text(f"Q{clip.period}  {int(gc // 60)}:{int(gc % 60):02d}")
            t_speed.set_text(f"{sp:.1f} ft/s")
            t_speed2.set_text(f"{sp * 0.3048:.1f} m/s  |  {sp * 0.6818:.1f} mph")
            minutes = max(t[i], 1e-6) / 60
            pace = dist_cum[i] / minutes if t[i] >= 5 else np.nan
            tval[0].set_text(f"{dist_cum[i]:.0f}")
            tval[1].set_text(f"{pace:.0f}" if np.isfinite(pace) else "...")
            tval[2].set_text(f"{eff_cum[i]}")
            if np.isfinite(pace):
                diff = (pace / clip.game_avg_pace - 1) * 100
                t_vsavg.set_text(f"avg {clip.game_avg_pace:.0f}  ({diff:+.0f}%)")
            trace.set_data(t[: i + 1], v[: i + 1])
            ei = np.flatnonzero(eff_start[: i + 1])
            eff_dots.set_offsets(np.column_stack([t[ei], v[ei]]) if len(ei) else np.empty((0, 2)))
            eff_dots.set_facecolors([INK if acc_start[k] else INK2 for k in ei])
            cursor.set_xdata([t[i], t[i]])
            # pills for active actions
            for p in pills:
                p.remove()
            pills.clear()
            active = spans[(spans["show_start"] <= f) & (spans["show_end"] >= f)]
            x = 0.065
            renderer = fig.canvas.get_renderer()
            for r in active.itertuples():
                pill = _pill(fig, x, 0.165, r.label, OPP if r.defense else TEAM)
                w = pill.get_window_extent(renderer).width / fig.bbox.width
                if x + w > 0.64:
                    pill.remove()
                    break
                pills.append(pill)
                x += w + 0.018
            done = spans[spans["show_start"] <= f]
            for k, row in enumerate(log_rows):
                if k < len(done):
                    r = done.iloc[k]
                    row.set_text(f"{'D' if r.defense else 'O'}   {r.short}")
                    row.set_color(INK if (r.show_end >= f) else INK2)
                else:
                    row.set_text("")
            writer.grab_frame()

        # ---------- outro ----------
        pace_full = dist_cum[-1]
        d_pct = (pace_full / clip.game_avg_pace - 1) * 100
        eff_pm = eff_cum[-1]
        n_def = int(spans["defense"].sum())
        n_off = len(spans) - n_def
        card([(0.80, "That minute, in numbers", 38, INK, True),
              (0.66, f"{pace_full:.0f} ft covered  ({pace_full * 0.3048:.0f} m)", 30, INK, True),
              (0.60, f"{d_pct:+.0f}% above his own game average of {clip.game_avg_pace:.0f} ft/min", 20, INK2, False),
              (0.50, f"{eff_pm} hard accelerations and decelerations in one minute  "
                     f"({eff_pm / clip.game_avg_efforts:.1f}x his game rate)", 22, INK, False),
              (0.43, f"Top speed {np.nanmax(v):.1f} ft/s  ({np.nanmax(v) * 0.3048:.1f} m/s)", 22, INK, False),
              (0.36, f"{n_def} defensive and {n_off} offensive actions: the load comes from what he was asked to do",
               20, INK2, False),
              (0.22, "Train the peak, not the average.", 26, FOCUS, True),
              (0.10, "Data: SkillCorner Open Data (Liga ACB 2025/26)  |  Chris Cothern, waims-tracking-lab", 13, INK3,
               False)])
        for _ in range(int(outro_s * FPS)):
            writer.grab_frame()
    plt.close(fig)
    plt.close(cfig)
    return out

