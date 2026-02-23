"""
app.py — Volleyball Analyzer Frontend
Clean, professional Streamlit interface with:
  - Video upload
  - Live progress tracking
  - Annotated video playback
  - Interactive filters (by action, player, team, time range)
  - Timeline + statistics charts
"""

import time
import httpx
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://api:8000"

# ── Action metadata ───────────────────────────────────────────────────────────
ACTION_META = {
    "serve":  {"label": "Serve",   "icon": "🏐", "color": "#6366f1"},
    "pass":   {"label": "Pass",    "icon": "🤝", "color": "#22c55e"},
    "set":    {"label": "Set",     "icon": "🙌", "color": "#f59e0b"},
    "attack": {"label": "Attack",  "icon": "⚡", "color": "#ef4444"},
    "block":  {"label": "Block",   "icon": "🚧", "color": "#8b5cf6"},
    "dig":    {"label": "Dig",     "icon": "🤸", "color": "#06b6d4"},
    "other":  {"label": "Other",   "icon": "❓", "color": "#9ca3af"},
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volleyball Analyzer",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    text-align: center;
    margin-bottom: 2rem;
  }
  .hero h1 { font-size: 2.4rem; font-weight: 800; color: #f8fafc; margin: 0; }
  .hero p  { color: #94a3b8; margin-top: .5rem; font-size: 1.05rem; }

  .stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .stat-card .val { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
  .stat-card .lbl { font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-top: .2rem; }

  .event-row {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: .9rem 1.2rem;
    margin-bottom: .5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    transition: border-color .15s;
  }
  .event-row:hover { border-color: #6366f1; }
  .event-time { font-size: .75rem; color: #64748b; font-variant-numeric: tabular-nums; min-width: 70px; }
  .event-badge {
    border-radius: 6px;
    padding: 3px 10px;
    font-size: .78rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .event-desc { font-size: .82rem; color: #94a3b8; flex: 1; }
  .event-conf { font-size: .75rem; color: #475569; min-width: 60px; text-align: right; }

  .upload-zone {
    border: 2px dashed #334155;
    border-radius: 14px;
    padding: 3rem;
    text-align: center;
    background: #0f172a;
  }

  .filter-chip {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: .78rem;
    font-weight: 600;
    margin: 3px;
    cursor: pointer;
    border: 2px solid transparent;
  }

  div[data-testid="stProgress"] > div { background-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m, s = int(seconds) // 60, seconds % 60
    return f"{m:02d}:{s:05.2f}"


def action_badge(action: str) -> str:
    meta = ACTION_META.get(action, ACTION_META["other"])
    return (
        f'<span class="event-badge" '
        f'style="background:{meta["color"]}22;color:{meta["color"]};'
        f'border:1px solid {meta["color"]}44">'
        f'{meta["icon"]} {meta["label"]}</span>'
    )


def poll_status(job_id: str) -> dict:
    try:
        r = httpx.get(f"{API_URL}/status/{job_id}", timeout=10)
        return r.json()
    except Exception:
        return {"status": "error", "message": "API unreachable"}


def events_to_df(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        meta = ACTION_META.get(ev["action_type"], ACTION_META["other"])
        rows.append({
            "Action":     ev["action_type"],
            "Label":      meta["label"],
            "Icon":       meta["icon"],
            "Color":      meta["color"],
            "Start (s)":  round(ev["start_time"], 2),
            "End (s)":    round(ev["end_time"], 2),
            "Duration":   round(ev["end_time"] - ev["start_time"], 2),
            "Confidence": round(ev["confidence"], 2),
            "Start fmt":  fmt_time(ev["start_time"]),
            "End fmt":    fmt_time(ev["end_time"]),
        })
    return pd.DataFrame(rows)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("job_id", None), ("result", None), ("polling", False),
    ("selected_actions", list(ACTION_META.keys())),
    ("selected_players", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏐 Volleyball Analyzer</h1>
  <p>AI-powered action detection · Player tracking · Team analysis</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result is None:

    st.markdown("### 📤 Upload a match video")
    uploaded = st.file_uploader(
        "Drop your video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            size_mb = len(uploaded.getvalue()) / 1024**2
            st.info(f"**{uploaded.name}** — {size_mb:.1f} MB")
        with col_btn:
            analyze = st.button("🚀 Analyze", use_container_width=True, type="primary")

        if analyze:
            with st.spinner("Uploading video..."):
                try:
                    r = httpx.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.session_state.job_id  = data["job_id"]
                    st.session_state.polling = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PROCESSING PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.polling and st.session_state.result is None:

    st.markdown("### ⚙️ Analyzing your video...")

    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)

    steps = [
        (step_col1, "📥 Download",   0.05),
        (step_col2, "🤖 Detection",  0.50),
        (step_col3, "🎯 Tracking",   0.80),
        (step_col4, "✅ Finalizing", 0.95),
    ]

    while True:
        status = poll_status(st.session_state.job_id)
        progress = status.get("progress", 0.0)
        progress_bar.progress(min(progress, 1.0))
        status_text.markdown(f"*{status.get('message', 'Processing...')}*")

        for col, label, threshold in steps:
            icon = "✅" if progress >= threshold else "⏳"
            col.markdown(f"**{icon} {label}**")

        if status["status"] == "done":
            st.session_state.result  = status.get("result", {})
            st.session_state.polling = False
            st.success("✅ Analysis complete!")
            time.sleep(0.5)
            st.rerun()
        elif status["status"] == "error":
            st.error(f"❌ Error: {status.get('message', 'Unknown error')}")
            if st.button("Try again"):
                st.session_state.job_id  = None
                st.session_state.polling = False
            break

        time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result:
    result = st.session_state.result
    events = result.get("events", [])
    df     = events_to_df(events)

    # ── Top stats row ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Match Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        (c1, len(events),                          "Total Events"),
        (c2, fmt_time(result.get("duration", 0)),  "Duration"),
        (c3, result.get("fps", 0),                 "FPS"),
        (c4, result.get("team_a_color","?").title(),"Team A Jersey"),
        (c5, result.get("team_b_color","?").title(),"Team B Jersey"),
    ]
    for col, val, lbl in stats:
        col.markdown(
            f'<div class="stat-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Main layout: video left, filters+events right ────────────────────────
    vid_col, filter_col = st.columns([3, 2], gap="large")

    with vid_col:
        st.markdown("#### 🎬 Annotated Video")
        try:
            r = httpx.get(f"{API_URL}/video/{st.session_state.job_id}", timeout=30)
            if r.status_code == 200:
                st.video(r.content)
            else:
                st.warning("Annotated video not available yet.")
        except Exception:
            st.info("Video stream not available in this environment.")

        # Timeline chart
        if not df.empty:
            st.markdown("#### 📅 Event Timeline")
            fig = px.timeline(
                df,
                x_start="Start (s)", x_end="End (s)",
                y="Label", color="Action",
                color_discrete_map={a: ACTION_META[a]["color"] for a in ACTION_META},
                height=280,
            )
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                xaxis_title="Time (seconds)", yaxis_title="",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            fig.update_xaxes(gridcolor="#1e293b")
            fig.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

    with filter_col:
        st.markdown("#### 🔍 Filters")

        # Action type filter
        st.markdown("**Action types**")
        action_counts = df["Action"].value_counts().to_dict() if not df.empty else {}
        selected = st.session_state.selected_actions.copy()

        cols = st.columns(3)
        for i, (action, meta) in enumerate(ACTION_META.items()):
            count = action_counts.get(action, 0)
            toggled = cols[i % 3].checkbox(
                f"{meta['icon']} {meta['label']} ({count})",
                value=(action in selected),
                key=f"chk_{action}",
            )
            if toggled and action not in selected:
                selected.append(action)
            elif not toggled and action in selected:
                selected.remove(action)

        st.session_state.selected_actions = selected

        # Time range filter
        st.markdown("**Time range**")
        duration = result.get("duration", 0)
        if duration > 0:
            t_min, t_max = st.slider(
                "Select interval (seconds)",
                0.0, float(duration),
                (0.0, float(duration)),
                step=0.5,
                label_visibility="collapsed",
            )
        else:
            t_min, t_max = 0.0, 0.0

        # Confidence filter
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)

        st.divider()

        # ── Filtered event list ───────────────────────────────────────
        st.markdown("#### 📋 Events")

        if df.empty:
            st.info("No events detected in this video.")
        else:
            filtered = df[
                df["Action"].isin(selected) &
                (df["Start (s)"] >= t_min) &
                (df["End (s)"] <= t_max) &
                (df["Confidence"] >= min_conf)
            ].reset_index(drop=True)

            st.caption(f"{len(filtered)} events shown")

            for _, row in filtered.iterrows():
                badge = action_badge(row["Action"])
                st.markdown(f"""
<div class="event-row">
  <span class="event-time">⏱ {row['Start fmt']}</span>
  {badge}
  <span class="event-conf">{int(row['Confidence']*100)}%</span>
  <span class="event-desc">{round(row['Duration'],1)}s</span>
</div>
                """, unsafe_allow_html=True)

        # ── Action breakdown chart ────────────────────────────────────
        st.markdown("#### 📈 Action Breakdown")
        if not df.empty:
            counts = df[df["Action"].isin(selected)]["Action"].value_counts()
            fig2 = go.Figure(go.Bar(
                x=[ACTION_META[a]["label"] for a in counts.index],
                y=counts.values,
                marker_color=[ACTION_META[a]["color"] for a in counts.index],
                text=counts.values,
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=30),
                height=220,
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Reset button ──────────────────────────────────────────────────────────
    st.divider()
    if st.button("📤 Analyze another video", use_container_width=False):
        for key in ["job_id", "result", "polling"]:
            st.session_state[key] = None if key != "polling" else False
        st.rerun()
