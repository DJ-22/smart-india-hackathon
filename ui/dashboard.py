import streamlit as st
import requests
import plotly.graph_objects as go

from risk import (
    RiskEngine,
    L1_AMBER,
    L1_RED,
)


# =========================================================
# CONFIG
# =========================================================

BACKEND_URL = "http://localhost:8000"

# Temporary session identifier.
# This is not detection data.
SESSION_ID = "demo-session"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VoiceGuard",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = RiskEngine()

if "windows" not in st.session_state:
    st.session_state.windows = []

if "last_window_id" not in st.session_state:
    st.session_state.last_window_id = -1

if "backend_connected" not in st.session_state:
    st.session_state.backend_connected = False

if "backend_returned_empty" not in st.session_state:
    st.session_state.backend_returned_empty = False

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# =========================================================
# BACKEND
# =========================================================

def get_windows():
    """
    Fetch newly processed windows from Person B's backend.

    Backend contract:

        GET /session/{sid}/windows?since={window_id}

    Returns:

        {
            "windows": [...]
        }

    No detection data is invented here.
    """

    try:
        response = requests.get(
            f"{BACKEND_URL}/session/{SESSION_ID}/windows",
            params={"since": st.session_state.last_window_id},
            timeout=1,
        )

        response.raise_for_status()

        data = response.json()

        windows = data.get("windows", [])

        st.session_state.backend_connected = True

        return windows

    except requests.RequestException:
        st.session_state.backend_connected = False
        return []


# =========================================================
# PROCESS WINDOWS
# =========================================================

def process_windows(windows):
    """
    Pass backend WindowScore objects through the risk engine.

    The backend supplies raw prob_fake.
    The risk engine determines the displayed risk.
    """

    for window in windows:

        st.session_state.windows.append(window)

        st.session_state.last_window_id = max(
            st.session_state.last_window_id,
            window["window_id"]
        )

        # Context currently comes from UI controls.
        context = {
            "unknown_caller": st.session_state.get(
                "unknown_caller",
                False
            ),
            "high_value_txn": st.session_state.get(
                "high_value_txn",
                False
            ),
            "off_hours": st.session_state.get(
                "off_hours",
                False
            ),
            "first_contact": st.session_state.get(
                "first_contact",
                False
            ),
        }

        st.session_state.risk_engine.update(
            window,
            context
        )


# =========================================================
# FETCH DATA
# =========================================================

new_windows = get_windows()

if new_windows:
    process_windows(new_windows)


# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("🛡️ VoiceGuard")

with header_right:
    if st.session_state.backend_connected:
        st.success("● BACKEND ONLINE")
    else:
        st.warning("● WAITING FOR BACKEND")


st.divider()


# =========================================================
# CALL STATUS
# =========================================================

status_left, status_middle, status_right = st.columns(3)

with status_left:
    if st.session_state.backend_connected:
        st.success("● LIVE")
    else:
        st.warning("● WAITING")

with status_middle:
    st.write(f"**Session:** `{SESSION_ID}`")

with status_right:
    if st.session_state.windows:
        latest = st.session_state.windows[-1]

        duration = latest["t_end"]

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        st.write(
            f"**Call duration:** "
            f"{minutes:02d}:{seconds:02d}"
        )
    else:
        st.write("**Call duration:** —")


# =========================================================
# NO DATA STATE
# =========================================================

if not st.session_state.windows:

    st.info(
        "Waiting for audio analysis from the detection backend."
    )

    st.caption(
        "No detection results are displayed until the backend "
        "returns WindowScore data."
    )

    # Still show context controls because these are
    # inputs to the risk engine, not fabricated detection data.


# =========================================================
# CONTEXT
# =========================================================

st.divider()

st.subheader("Risk Context")

ctx1, ctx2, ctx3, ctx4 = st.columns(4)

with ctx1:
    st.checkbox(
        "Unknown caller ID",
        key="unknown_caller",
    )

with ctx2:
    st.checkbox(
        "High-value transaction",
        key="high_value_txn",
    )

with ctx3:
    st.checkbox(
        "Outside business hours",
        key="off_hours",
    )

with ctx4:
    st.checkbox(
        "First contact",
        key="first_contact",
    )


# =========================================================
# RISK
# =========================================================

st.divider()

left, right = st.columns([1, 2])


# =========================================================
# CURRENT RISK
# =========================================================

with left:

    st.subheader("Current Risk")

    risk_state = st.session_state.risk_engine.state()

    if risk_state["risk"] is None:

        st.metric(
            "Risk",
            "—"
        )

        st.info(
            "Risk will appear after the first valid "
            "audio window is analysed."
        )

    else:

        risk = risk_state["risk"]
        band = risk_state["band"]

        st.metric(
            "Impersonation Risk",
            f"{risk:.2f}"
        )

        if band == "RED":
            st.error("🔴 RED RISK")

        elif band == "AMBER":
            st.warning("🟠 AMBER RISK")

        else:
            st.success("🟢 GREEN RISK")

        st.progress(
            min(max(risk, 0.0), 1.0)
        )


# =========================================================
# RISK GRAPH
# =========================================================

with right:

    st.subheader("Risk Over Time")

    history = st.session_state.risk_engine.history

    if not history:

        st.info(
            "Risk history will appear once audio windows "
            "are received from the backend."
        )

    else:

        x = [
            item["t_end"]
            for item in history
        ]

        y = [
            item["risk"]
            for item in history
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name="Risk",
            )
        )

        fig.add_hline(
            y=0.45,
            line_dash="dash",
            annotation_text="AMBER"
        )

        fig.add_hline(
            y=0.75,
            line_dash="dash",
            annotation_text="RED"
        )

        fig.update_layout(
            height=350,
            yaxis=dict(
                range=[0, 1],
                title="Risk"
            ),
            xaxis_title="Time (seconds)",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# DETECTION CASCADE
# =========================================================

st.divider()

st.subheader("Detection Cascade")

level_counts = {
    0: 0,
    1: 0,
    2: 0,
}

for window in st.session_state.windows:

    level = window.get("level_resolved")

    if level in level_counts:
        level_counts[level] += 1


total_windows = sum(level_counts.values())

c1, c2, c3 = st.columns(3)

with c1:

    if total_windows:
        percentage = level_counts[0] / total_windows
        value = f"{percentage:.0%}"
    else:
        value = "—"

    st.metric(
        "Level 0",
        value,
        "Screening"
    )

with c2:

    if total_windows:
        percentage = level_counts[1] / total_windows
        value = f"{percentage:.0%}"
    else:
        value = "—"

    st.metric(
        "Level 1",
        value,
        "Detection"
    )

with c3:

    if total_windows:
        percentage = level_counts[2] / total_windows
        value = f"{percentage:.0%}"
    else:
        value = "—"

    st.metric(
        "Level 2",
        value,
        "Attribution"
    )


# =========================================================
# LATEST WINDOW
# =========================================================

if st.session_state.windows:

    st.divider()

    st.subheader("Latest Detection Window")

    latest = st.session_state.windows[-1]

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.metric(
            "Window",
            latest["window_id"]
        )

    with w2:
        st.metric(
            "Fake Probability",
            f"{latest['prob_fake']:.2f}"
        )

    with w3:
        st.metric(
            "Speech Ratio",
            f"{latest['speech_ratio']:.0%}"
        )

    with w4:
        st.metric(
            "Latency",
            f"{latest['latency_ms']} ms"
        )


# =========================================================
# ALERT
# =========================================================

risk_state = st.session_state.risk_engine.state()

if risk_state["band"] == "RED":

    st.divider()

    st.error(
        """
        ⚠️ HIGH IMPERSONATION RISK

        The current risk level has reached RED.
        Secondary verification is recommended before
        approving a sensitive action.
        """
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        st.button("📞 Request Call-back")

    with a2:
        st.button("🚨 Escalate")

    with a3:
        st.button("Dismiss")


# =========================================================
# EVENT LOG
# =========================================================

st.divider()

st.subheader("Event Log")

events = st.session_state.risk_engine.events

if not events:

    st.info(
        "No risk events yet."
    )

else:

    for event in reversed(events):

        st.write(
            f"• {event}"
        )


# =========================================================
# AUTO REFRESH
# =========================================================

time.sleep(0.5)

st.rerun()