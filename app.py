"""Game Demands Lab: Streamlit app for waims-tracking-lab.

Runs only on the small committed tables in exports/waims and the clips in
assets/video, so it deploys to Streamlit Cloud without the raw tracking data.
Data: SkillCorner Open Data (MIT), Liga ACB 2025/26.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "exports" / "waims"
VIDEO = ROOT / "assets" / "video"
REPO = "https://github.com/dchriscothern/waims-tracking-lab"
FT_TO_M = 0.3048

ROLES = ["guard", "wing", "big"]
ROLE_COLOR_LIGHT = {"guard": "#2a78d6", "wing": "#eb6834", "big": "#1baf7a"}
ROLE_COLOR_DARK = {"guard": "#3987e5", "wing": "#d95926", "big": "#199e70"}

st.set_page_config(page_title="Game Demands Lab", page_icon="🏀", layout="wide")


# ---------- data ----------
@st.cache_data
def load():
    d = {n: pd.read_csv(DATA / f"{n}.csv") for n in
         ("game_demands_player", "role_benchmarks", "role_by_quarter", "action_cost",
          "pick_coverage", "peak_context", "drive_blowby")}
    pg = d["game_demands_player"]
    d["pg10"] = pg[(pg["minutes_running"] >= 10) & pg["role"].isin(ROLES)].copy()
    return d


D = load()
PG = D["pg10"]


def dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


ROLE_COLOR = ROLE_COLOR_DARK if dark_mode() else ROLE_COLOR_LIGHT

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Game Demands Lab")
    units = st.radio("Units", ["feet", "meters"], horizontal=True)
    st.caption("Pro game load by role and the physical cost of basketball actions, "
               "from broadcast tracking of 10 Liga ACB 2025/26 games.")
    st.markdown(f"[Code and methods on GitHub]({REPO})")
    st.caption("Data: SkillCorner Open Data (MIT). Built by Chris Cothern, PT, CSCS, CPSS.")

M = units == "meters"
DU = "m" if M else "ft"          # distance unit
K = FT_TO_M if M else 1.0         # multiply feet values by this


def dist(x):
    return x * K


def fmt(x, dec=0):
    return f"{x:,.{dec}f}"


def med(role, col):
    return PG.loc[PG["role"] == role, col].median()


# ---------- header ----------
st.title("What does a basketball action cost?")
st.markdown("Game physical demands by role, and the load cost of each tactical action, measured from "
            "**SkillCorner broadcast tracking** (Liga ACB 2025/26, 10 games, 25 frames per second). "
            "All values use running-clock time.")

tabs = st.tabs(["Overview", "Demands by role", "Action cost", "Compare your athlete",
                "Peak minute video", "Methods"])

# ---------- overview ----------
with tabs[0]:
    cols = st.columns(3)
    for c, role in zip(cols, ROLES):
        with c:
            st.markdown(f"#### {role.title()}s")
            n = int((PG["role"] == role).sum())
            a, b = st.columns(2)
            a.metric(f"Distance ({DU}/min)", fmt(dist(med(role, "distance_per_min"))))
            b.metric(f"Peak 60 s ({DU}/min)", fmt(dist(med(role, "peak60_dist_per_min"))))
            a.metric(f"High-speed ({DU}/min)", fmt(dist(med(role, "hsd_per_min")), 1))
            b.metric("Accels + decels / min", fmt(med(role, "efforts_per_min"), 2))
            st.caption(f"Median of {n} player-games with 10+ minutes")

    ratios = {r: med(r, "peak60_dist_per_min") / med(r, "distance_per_min") for r in ROLES}
    st.markdown("### Key findings")
    q = D["role_by_quarter"]
    drop = {r: 1 - q[(q.role == r) & (q.period == 4)].distance_per_min.iloc[0]
            / q[(q.role == r) & (q.period == 1)].distance_per_min.iloc[0] for r in ROLES}
    ac = D["action_cost"]

    def wmean(action, actor):
        s = ac[(ac.action == action) & (ac.actor == actor)]
        return (s.mean_excess_speed * s.n).sum() / s.n.sum()

    st.markdown(f"""
1. **Guards and wings cover the same ground, but differently.** Wings do the most high-speed running;
   guards do the most accelerating and braking.
2. **The peak minute runs about a third above the game average** for every role
   (guards {ratios['guard']:.2f}x, wings {ratios['wing']:.2f}x, bigs {ratios['big']:.2f}x).
   Conditioning to game averages underprepares players.
3. **Intensity fades by the fourth quarter:** distance per minute falls about {drop['wing']:.0%} for wings,
   {drop['big']:.0%} for bigs and {drop['guard']:.0%} for guards from Q1 to Q4.
4. **Drives are the costliest offensive action** (+{dist(wmean('drives', 'ballhandler')):.1f} {DU}/s above the
   player's own average). **Closeouts are the sharpest defensive cost** (+{dist(wmean('closeouts', 'ballhandler_def')):.1f} {DU}/s
   in about 1.3 s).
5. **Switching a pick costs both defenders less** than going over, under or showing.
""")

# ---------- demands by role ----------
with tabs[1]:
    metrics = {
        "Distance per minute": ("distance_per_min", True),
        "High-speed distance per minute (16+ ft/s, 4.9+ m/s)": ("hsd_per_min", True),
        "Accels + decels per minute (above 3 m/s²)": ("efforts_per_min", False),
        "Peak 60 s distance per minute": ("peak60_dist_per_min", True),
        "Peak 30 s distance per minute": ("peak30_dist_per_min", True),
        "Top speed": ("peak_speed", True),
        "Sprints per game (20+ ft/s for 0.5 s)": ("sprints", False),
    }
    choice = st.selectbox("Metric", list(metrics))
    col, is_dist = metrics[choice]
    unit = {"peak_speed": f"{DU}/s", "sprints": "count", "efforts_per_min": "per min"}.get(col, f"{DU}/min")
    d = PG.assign(value=PG[col] * (K if is_dist else 1))
    fig = px.strip(d, x="role", y="value", color="role", category_orders={"role": ROLES},
                   color_discrete_map=ROLE_COLOR, hover_data={"team_name": True, "minutes_running": ":.1f",
                                                              "role": False, "value": ":.2f"})
    for i, r in enumerate(ROLES):
        m = d.loc[d.role == r, "value"].median()
        fig.add_shape(type="line", x0=i - 0.3, x1=i + 0.3, y0=m, y1=m, line=dict(width=3))
        fig.add_annotation(x=i + 0.32, y=m, text=f"median {m:.1f}", showarrow=False, xanchor="left")
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(showlegend=False, height=430, yaxis_title=unit, xaxis_title=None,
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption("Each dot is one player-game with 10+ running-clock minutes. Hover for team and minutes.")

    st.markdown("#### Intensity by quarter")
    q = D["role_by_quarter"][D["role_by_quarter"].period <= 4]
    c1, c2 = st.columns(2)
    for c, (qcol, lab, isd) in zip((c1, c2), [("distance_per_min", f"Distance ({DU}/min)", True),
                                             ("efforts_per_min", "Accels + decels per min", False)]):
        f = px.line(q.assign(v=q[qcol] * (K if isd else 1), Q="Q" + q.period.astype(str)), x="Q", y="v",
                    color="role", markers=True, category_orders={"role": ROLES}, color_discrete_map=ROLE_COLOR)
        f.update_layout(height=320, yaxis_title=lab, xaxis_title=None, legend_title=None, margin=dict(t=10))
        c.plotly_chart(f, width="stretch", theme="streamlit")

    st.markdown("#### Pro benchmark bands")
    b = D["role_benchmarks"].copy()
    names = {"distance_per_min": f"Distance ({DU}/min)", "hsd_per_min": f"High-speed ({DU}/min)",
             "efforts_per_min": "Accels + decels / min", "peak_speed": f"Top speed ({DU}/s)",
             "peak30_dist_per_min": f"Peak 30 s ({DU}/min)", "peak60_dist_per_min": f"Peak 60 s ({DU}/min)",
             "sprints": "Sprints", "distance_ft": f"Game distance ({DU})"}
    distcols = {"distance_per_min", "hsd_per_min", "peak_speed", "peak30_dist_per_min",
                "peak60_dist_per_min", "distance_ft"}
    for c in ("p10", "p50", "p90"):
        b[c] = np.where(b.metric.isin(distcols), b[c] * K, b[c])
    b["metric"] = b.metric.map(names)
    t = b.pivot_table(index="metric", columns="role", values="p50")[ROLES]
    band = b.assign(txt=lambda x: x.p10.round(1).astype(str) + " to " + x.p90.round(1).astype(str)) \
        .pivot_table(index="metric", columns="role", values="txt", aggfunc="first")[ROLES]
    show = t.round(1).astype(str) + "  (" + band + ")"
    st.dataframe(show.rename(columns=str.title), width="stretch")
    st.caption("Median, with the 10th to 90th percentile range in brackets.")

# ---------- action cost ----------
with tabs[2]:
    st.markdown("Speed inside each action window minus the **same player's own game average**. "
                "Positive means the action is harder than that player's normal running.")
    role_pick = st.radio("Player role", ["all"] + ROLES, horizontal=True)
    ac = D["action_cost"] if role_pick == "all" else D["action_cost"][D["action_cost"].player_role == role_pick]
    w = ac.assign(wx=ac.mean_excess_speed * ac.n).groupby(["action", "actor"]).agg(wx=("wx", "sum"), n=("n", "sum"))
    w["excess"] = w.wx / w.n * K
    w = w.reset_index()
    w["side"] = np.where(w.actor.str.endswith("_def"), "Their defender", "Player with the action")
    w["who"] = w.action.str.replace("_", " ") + ": " + w.actor.str.replace("_def", "", regex=False)
    order = ["drives", "closeouts", "isolations", "posts", "picks", "handoffs", "off_ball_screens"]
    w["o"] = w.action.map({a: i for i, a in enumerate(order)})
    piv = w.sort_values(["o", "who"]).pivot_table(index="who", columns="side", values="excess", sort=False)
    npiv = w.pivot_table(index="who", columns="side", values="n", sort=False).reindex(piv.index)
    piv = piv[["Player with the action", "Their defender"]]
    npiv = npiv[piv.columns]
    lim = float(np.nanmax(np.abs(piv.values)))
    hm = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
        text=[[f"{v:+.1f}  (n={int(n)})" if np.isfinite(v) else "" for v, n in zip(r, nr)]
              for r, nr in zip(piv.values, npiv.fillna(0).values)],
        texttemplate="%{text}", textfont=dict(color="#0b0b0b"),
        colorbar=dict(title=f"{DU}/s"),
        hovertemplate="%{y}<br>%{x}: %{z:+.2f} " + DU + "/s<extra></extra>"))
    hm.update_layout(height=520, yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(hm, width="stretch", theme="streamlit")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pick coverage and defender workload")
        pc = D["pick_coverage"].assign(
            x=lambda d: d.mean_excess_speed * K, e=lambda d: d.ci95 * K,
            who=lambda d: d.actor.map({"ballhandler_def": "Ball handler's defender",
                                       "screener_def": "Screener's defender"}),
            label=lambda d: d.coverage + " (n=" + d.n.astype(str) + ")")
        f = px.scatter(pc, x="x", y="label", error_x="e", facet_row="who",
                       color_discrete_sequence=[ROLE_COLOR["guard"]])
        f.add_vline(x=0, line_width=1)
        f.update_traces(marker=dict(size=11))
        f.update_yaxes(matches=None, title=None)
        f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        f.update_layout(height=420, xaxis_title=f"Excess speed ({DU}/s), 95% CI", margin=dict(t=30))
        st.plotly_chart(f, width="stretch", theme="streamlit")
        st.caption("Switching costs both defenders less than the alternatives.")
    with c2:
        st.markdown("#### What fills the most intense minute")
        pk = D["peak_context"].copy()
        pk["label"] = pk.action.str.replace("_", " ") + np.where(pk.actor_side == "def", " (defending)", "")
        keep = pk.groupby("label").per_window.max().nlargest(8).index
        pk = pk[pk.label.isin(keep)]
        f = px.bar(pk, y="label", x="ratio_vs_overall", color="role", barmode="group", orientation="h",
                   category_orders={"role": ROLES}, color_discrete_map=ROLE_COLOR,
                   hover_data={"per_window": ":.2f", "per_minute_overall": ":.2f"})
        f.add_vline(x=1, line_width=1, line_dash="dot")
        f.update_layout(height=420, xaxis_title="Times more often than in an average minute",
                        yaxis_title=None, legend_title=None, margin=dict(t=30))
        st.plotly_chart(f, width="stretch", theme="streamlit")
        st.caption("Above 1 means the action shows up more in the player's peak 60 s than in an average minute.")

# ---------- compare your athlete ----------
with tabs[3]:
    st.markdown("Enter one of your athlete's game values and see where it sits against **pro players in the "
                "same role**. Use running-clock (live ball) minutes, and the same thresholds as here: "
                "high speed from 16 ft/s (4.9 m/s), efforts above 3 m/s² for at least 0.3 s.")
    role = st.selectbox("Athlete's role", ROLES, format_func=str.title)
    ref = PG[PG.role == role]
    spec = [("distance_per_min", f"Distance per minute ({DU}/min)", True),
            ("hsd_per_min", f"High-speed distance per minute ({DU}/min)", True),
            ("efforts_per_min", "Accels + decels per minute", False),
            ("peak60_dist_per_min", f"Peak 60 s distance per minute ({DU}/min)", True)]
    cols = st.columns(4)
    vals = {}
    for c, (k, lab, isd) in zip(cols, spec):
        default = float(ref[k].median() * (K if isd else 1))
        vals[k] = c.number_input(lab, min_value=0.0, value=round(default, 1),
                                 step=1.0 if isd else 0.1, key=f"{k}_{units}_{role}")
    st.markdown("---")
    for k, lab, isd in spec:
        dist_vals = ref[k].dropna() * (K if isd else 1)
        v = vals[k]
        pct = (dist_vals < v).mean() * 100
        p10, p50 = dist_vals.quantile(0.1), dist_vals.median()
        if v >= p50:
            dot, status = "🟢", "AT PRO LEVEL"
        elif v >= p10:
            dot, status = "🟡", "BELOW PRO MEDIAN"
        else:
            dot, status = "🔴", "WELL BELOW PRO"
        a, b = st.columns([2, 3])
        a.markdown(f"**{lab}**  \n{dot} {status}")
        b.progress(min(max(pct / 100, 0.0), 1.0),
                   text=f"{v:.1f} is higher than {pct:.0f}% of pro {role} games (pro median {p50:.1f})")
    st.caption("Pro reference: player-games with 10+ running-clock minutes in the 10 published games. "
               "Broadcast tracking differs from wearable GPS or LPS, so treat small differences with caution.")

# ---------- video ----------
with tabs[4]:
    clips = {
        "Guard: BC Barcelona, Q3 vs Gran Canaria": ("peak_minute_guard.mp4",
            "484 ft in the minute, 31% above his game pace. Ten hard accelerations and decelerations (2.4x his game "
            "rate). He chases through two screens, guards an isolation and a drive, then runs a handoff, a pick and "
            "roll, a cut and a drive to the rim."),
        "Wing: Casademont Zaragoza, Q1 vs Breogan": ("peak_minute_wing.mp4",
            "598 ft in the minute, 40% above his game pace, with nine hard efforts. Mostly defense: chasing through a "
            "screen, guarding the ball in a pick and a drive, then a cut off a screen and a handoff."),
    }
    pick = st.radio("Clip", list(clips), horizontal=True)
    f, cap = clips[pick]
    st.video(str(VIDEO / f))
    st.caption(cap)
    st.caption("2D replay rendered from tracking data (no broadcast video). Hollow dots are positions estimated "
               "off-camera. Each clip: intro card, the peak 60 s in real time, summary card.")

# ---------- methods ----------
with tabs[5]:
    st.markdown(f"""
**Data.** SkillCorner Open Data, Liga ACB 2025/26: 10 games of broadcast tracking (player and ball positions at
25 fps) plus Dynamic Events (picks, drives, closeouts and more). About 83% of positions are detected on camera;
the rest are estimated.

**Cleaning.** Provided speed (it matches position-derived speed, r = 0.998) is capped at 26 ft/s and smoothed with
a Savitzky Golay filter (0.52 s window). Acceleration is the change in smoothed speed.

**Metrics.** Running-clock time only. High speed means 16+ ft/s; a sprint means 20+ ft/s for 0.5 s; an effort is
an acceleration or deceleration above 9.8 ft/s² (3 m/s²) for 0.3 s. Peak windows are the best 30 s and 60 s of
running clock within a period.

**Roles.** Positions are not in the data, so guard, wing and big come from what players do on court: touches,
drives, picks as handler or screener, off-ball screens, posts, rebounds, shot mix and touch distance from the rim.
A clustering method agrees with the rule on 80% of players.

**Action cost.** Speed inside each action window minus the player's own game average, which controls for players
who are always fast.

**Limitations.** Broadcast tracking (about 17% estimated positions); locomotion only, so no jump or contact load;
effort counts depend on the filter (up to 14%); 10 games of ACB men's basketball, so transfer to other leagues is
an assumption.

Full methods, validation and code: [{REPO}]({REPO})
""")
