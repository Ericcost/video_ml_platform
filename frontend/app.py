"""
app.py — Volleyball Analyzer Frontend  v2
Fixes:
  - Vidéo annotée : lue depuis MinIO en bytes et passée à st.video()
  - Timeline : utilise go.Bar horizontal (px.timeline nécessite des dates)
  - Gestion propre du cas "0 events"
"""

import time
import httpx
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://api:8000"

ACTION_META = {
    "serve":  {"label": "Serve",  "icon": "🏐", "color": "#6366f1"},
    "pass":   {"label": "Pass",   "icon": "🤝", "color": "#22c55e"},
    "set":    {"label": "Set",    "icon": "🙌", "color": "#f59e0b"},
    "attack": {"label": "Attack", "icon": "⚡", "color": "#ef4444"},
    "block":  {"label": "Block",  "icon": "🚧", "color": "#8b5cf6"},
    "dig":    {"label": "Dig",    "icon": "🤸", "color": "#06b6d4"},
    "other":  {"label": "Other",  "icon": "❓", "color": "#9ca3af"},
}

st.set_page_config(
    page_title="Volleyball Analyzer",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155; border-radius: 16px;
    padding: 2rem 3rem; text-align: center; margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin: 0; }
  .hero p  { color: #94a3b8; margin-top: .4rem; }

  .stat-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.1rem 1.4rem; text-align: center;
  }
  .stat-card .val { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
  .stat-card .lbl { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }

  .event-row {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: .8rem 1.1rem; margin-bottom: .4rem;
    display: flex; align-items: center; gap: .8rem;
  }
  .event-time  { font-size: .75rem; color: #64748b; min-width: 65px; font-variant-numeric: tabular-nums; }
  .event-badge { border-radius: 6px; padding: 2px 9px; font-size: .75rem; font-weight: 600; white-space: nowrap; }
  .event-dur   { font-size: .75rem; color: #475569; margin-left: auto; }

  .disclaimer  {
    background: #fef3c7; border: 1px solid #f59e0b;
    border-radius: 8px; padding: .6rem 1rem; font-size: .78rem; color: #92400e; margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_time(s: float) -> str:
    return f"{int(s)//60:02d}:{s%60:05.2f}"

def action_badge(action: str) -> str:
    m = ACTION_META.get(action, ACTION_META["other"])
    return (f'<span class="event-badge" '
            f'style="background:{m["color"]}22;color:{m["color"]};border:1px solid {m["color"]}44">'
            f'{m["icon"]} {m["label"]}</span>')

def poll(job_id: str) -> dict:
    try:
        return httpx.get(f"{API_URL}/status/{job_id}", timeout=10).json()
    except Exception:
        return {"status": "error", "message": "API unreachable"}

def events_to_df(events):
    rows = []
    for ev in events:
        a = ev["action_type"]
        m = ACTION_META.get(a, ACTION_META["other"])
        rows.append({
            "Action":    a,
            "Label":     m["label"],
            "Color":     m["color"],
            "Start":     round(ev["start_time"], 2),
            "End":       round(ev["end_time"],   2),
            "Duration":  round(ev["end_time"] - ev["start_time"], 2),
            "Conf":      round(ev.get("confidence", 0), 2),
            "StartFmt":  fmt_time(ev["start_time"]),
        })
    return pd.DataFrame(rows)


# ── session state ─────────────────────────────────────────────────────────────

for k, v in [("job_id", None), ("result", None), ("polling", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🏐 Volleyball Analyzer</h1>
  <p>AI-powered action detection · Player tracking · Team analysis</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result is None and not st.session_state.polling:

    st.markdown("### 📤 Upload a match video")
    uploaded = st.file_uploader(
        "Drop your video here", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        col_info.info(f"**{uploaded.name}** — {len(uploaded.getvalue())/1024**2:.1f} MB")

        if col_btn.button("🚀 Analyze", use_container_width=True, type="primary"):
            with st.spinner("Uploading..."):
                try:
                    r = httpx.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                        timeout=120,
                    )
                    r.raise_for_status()
                    st.session_state.job_id  = r.json()["job_id"]
                    st.session_state.polling = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS POLLING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.polling and st.session_state.result is None:

    st.markdown("### ⚙️ Analyzing your video...")
    bar  = st.progress(0.0)
    info = st.empty()

    STEPS = [
        (0.05, "📥 Download"),
        (0.50, "🤖 Detection"),
        (0.85, "🎯 Tracking"),
        (0.95, "✅ Finalizing"),
    ]

    # One st.empty() per step — allows in-place updates (no duplication)
    cols = st.columns(len(STEPS))
    step_slots = []
    for col, (_, label) in zip(cols, STEPS):
        with col:
            slot = st.empty()
            slot.markdown(f"⏳ **{label}**")
            step_slots.append(slot)

    steps_done  = [False] * len(STEPS)
    err_streak  = 0          # consecutive poll failures
    MAX_ERRORS  = 5          # tolerate brief network blips (e.g. during ffmpeg)

    while True:
        status = poll(st.session_state.job_id)

        if status["status"] == "error":
            err_streak += 1
            if err_streak >= MAX_ERRORS:
                st.error(f"❌ {status.get('message', 'Unknown error')}")
                if st.button("↩ Try again"):
                    st.session_state.polling = False
                break
            time.sleep(2)
            continue

        err_streak = 0
        progress   = float(status.get("progress", 0.0))

        bar.progress(min(progress, 1.0))
        info.markdown(f"*{status.get('message', 'Processing...')}*")

        # Mark a step done only the first time its threshold is crossed
        for i, (thr, label) in enumerate(STEPS):
            if not steps_done[i] and progress >= thr:
                steps_done[i] = True
                step_slots[i].markdown(f"✅ **{label}**")

        if status["status"] == "done":
            st.session_state.result  = status.get("result", {})
            st.session_state.polling = False
            st.rerun()

        time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result:
    result = st.session_state.result
    events = result.get("events", [])
    df     = events_to_df(events)

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Match Overview")
    cols = st.columns(5)
    for col, val, lbl in [
        (cols[0], len(events),                            "Total Events"),
        (cols[1], fmt_time(result.get("duration", 0)),   "Duration"),
        (cols[2], f"{result.get('fps', 0):.0f}",         "FPS"),
        (cols[3], result.get("team_a_color", "?").title(),"Team A"),
        (cols[4], result.get("team_b_color", "?").title(),"Team B"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Main columns ──────────────────────────────────────────────────────────
    # filter_col is filled FIRST in code so `filtered` exists before vid_col
    # renders the timeline. Streamlit places columns left-to-right by creation
    # order (vid_col, filter_col), regardless of which `with` block runs first.
    vid_col, filter_col = st.columns([3, 2], gap="large")

    # ── FILTERS (compute filtered before drawing the timeline) ────────────────
    with filter_col:
        st.markdown("#### 🔍 Filters")

        action_counts = df["Action"].value_counts().to_dict() if not df.empty else {}
        selected = []
        grid = st.columns(3)
        for i, (action, meta) in enumerate(ACTION_META.items()):
            count = action_counts.get(action, 0)
            if grid[i % 3].checkbox(
                f"{meta['icon']} {meta['label']} ({count})",
                value=True,
                key=f"chk_{action}",
            ):
                selected.append(action)

        duration = float(result.get("duration", 0))
        if duration > 0:
            t_min, t_max = st.slider(
                "Time range (s)", 0.0, duration, (0.0, duration), step=0.5,
            )
        else:
            t_min, t_max = 0.0, 0.0

        min_conf = st.slider(
            "Min confidence — hide low-confidence events (classifier certainty 0→1)",
            0.0, 1.0, 0.4, 0.05,
        )

        # Compute filtered DataFrame here so vid_col can use it too
        if not df.empty:
            filtered = df[
                df["Action"].isin(selected) &
                (df["Start"] >= t_min) &
                (df["End"] <= t_max) &
                (df["Conf"] >= min_conf)
            ]
        else:
            filtered = df

        st.divider()
        st.markdown("#### 📋 Events")

        if filtered.empty:
            st.info("No events match the current filters.")
        else:
            st.caption(f"{len(filtered)} event(s) shown")
            for _, row in filtered.iterrows():
                st.markdown(
                    f'<div class="event-row">'
                    f'<span class="event-time">⏱ {row["StartFmt"]}</span>'
                    f'{action_badge(row["Action"])}'
                    f'<span class="event-dur">{row["Duration"]:.1f}s &nbsp; {int(row["Conf"]*100)}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── ACTION BREAKDOWN (uses filtered) ─────────────────────────────────
        st.markdown("#### 📈 Action Breakdown")
        if filtered.empty:
            st.info("No data for current filters.")
        else:
            counts = filtered["Action"].value_counts()
            fig2 = go.Figure(go.Bar(
                x=[ACTION_META.get(a, {}).get("label", a) for a in counts.index],
                y=counts.values,
                marker_color=[ACTION_META.get(a, {}).get("color", "#888") for a in counts.index],
                text=counts.values,
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8", showlegend=False,
                margin=dict(l=0, r=0, t=10, b=30), height=220,
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── VIDEO + TIMELINE (uses filtered for the timeline) ─────────────────────
    with vid_col:
        st.markdown("#### 🎬 Annotated Video")
        try:
            r = httpx.get(f"{API_URL}/video/{st.session_state.job_id}", timeout=60)
            if r.status_code == 200:
                st.video(r.content)
            else:
                st.info(f"Video not ready (HTTP {r.status_code}). "
                        f"Check MinIO at http://localhost:9001 "
                        f"(bucket: output-videos, login: volleyball / volleyball123)")
        except Exception as e:
            st.warning(f"Could not load video: {e}")

        st.markdown("#### 📅 Event Timeline")
        if filtered.empty:
            st.info("No events match the current filters.")
        else:
            fig = go.Figure()
            duration_total = result.get("duration", 1)

            for _, row in filtered.iterrows():
                m = ACTION_META.get(row["Action"], ACTION_META["other"])
                fig.add_trace(go.Bar(
                    name=m["label"],
                    x=[row["Duration"]],
                    y=[m["label"]],
                    base=[row["Start"]],
                    orientation="h",
                    marker_color=m["color"],
                    text=f"{row['StartFmt']}  {m['icon']}",
                    textposition="inside",
                    hovertemplate=(
                        f"<b>{m['label']}</b><br>"
                        f"Start: {row['StartFmt']}<br>"
                        f"Duration: {row['Duration']:.1f}s<br>"
                        f"Confidence: {int(row['Conf']*100)}%<extra></extra>"
                    ),
                ))

            fig.update_layout(
                barmode="overlay",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                xaxis=dict(title="Time (seconds)", range=[0, duration_total],
                           gridcolor="#1e293b"),
                yaxis=dict(title="", gridcolor="#1e293b"),
                showlegend=False,
                height=max(180, len(filtered["Action"].unique()) * 45 + 60),
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    if st.button("📤 Analyze another video"):
        for k in ["job_id", "result"]:
            st.session_state[k] = None
        st.session_state.polling = False
        st.rerun()
