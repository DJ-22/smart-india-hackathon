import os
import sys
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Repo root on sys.path FIRST, so `common` and `server` import regardless of the
# directory Streamlit is launched from (and before `risk`, which needs it too).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk import RiskEngine

# Bands come straight from the calibrated source, not re-exported through risk.
from common.config import L1_AMBER, L1_RED


# =========================================================
# CONFIG
# =========================================================

# Overridable so the server can move ports/hosts without a code change.
BACKEND_URL = os.getenv(
    "VG_SERVER_URL",
    "http://127.0.0.1:8000",
)

LIVEKIT_URL = os.getenv("LIVEKIT_URL")

# Which backend session to monitor. Overridable so the dashboard can point at a
# live identity (e.g. "daksh") without editing code; defaults to the demo id.
SESSION_ID = os.getenv("VG_SESSION_ID", "demo-session")

# Poll cadence. The backend emits one window per second (HOP_S = 1.0), so a 1 s
# poll matches the data rate exactly. Only the live fragment reruns at this rate.
POLL_INTERVAL_S = 1.0

# Consecutive connected-but-empty polls before we suspect the stream restarted
# (window_id resets to 0 on a server restart or /reset) and re-probe from -1.
EMPTY_RESET_TICKS = 5


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

# Consecutive polls that succeeded but returned no new windows.
if "empty_streak" not in st.session_state:
    st.session_state.empty_streak = 0


# =========================================================
# BACKEND
# =========================================================

def get_windows(since=None):
    """
    Fetch windows from the backend.

        GET /session/{sid}/windows?since={window_id}  ->  {"windows": [...]}

    since=None uses the persisted cursor (incremental fetch). Pass since=-1 to
    re-read from the beginning. Sets backend_connected as a side effect and
    never raises: a down/slow backend returns [] and flips the flag so the
    caller can keep the last known values.
    """

    if since is None:
        since = st.session_state.last_window_id

    try:
        response = requests.get(
            f"{BACKEND_URL}/session/{SESSION_ID}/windows",
            params={"since": since},
            timeout=2,
        )
        response.raise_for_status()
        st.session_state.backend_connected = True
        return response.json().get("windows", [])

    except requests.RequestException:
        st.session_state.backend_connected = False
        return []


def process_windows(windows):
    """
    Feed new backend WindowScores through the risk engine, in order.

    Each window is fed exactly once: anything at or below the cursor is skipped,
    so re-polling the boundary can never double-count a window into the EMA. The
    engine itself is incremental (one window per update) and drops gated windows
    (speech_ratio < MIN_SPEECH_RATIO) before they reach the EMA or clean streak.
    """

    for window in windows:

        wid = window["window_id"]

        if wid <= st.session_state.last_window_id:
            continue  # already seen -- never re-feed the smoother

        st.session_state.windows.append(window)
        st.session_state.last_window_id = wid

        context = {
            "unknown_caller": st.session_state.get("unknown_caller", False),
            "high_value_txn": st.session_state.get("high_value_txn", False),
            "off_hours": st.session_state.get("off_hours", False),
            "first_contact": st.session_state.get("first_contact", False),
        }

        st.session_state.risk_engine.update(window, context)


def _rebuild_if_stream_restarted():
    """
    Called after a run of connected-but-empty polls. window_id restarts at 0 on
    a server restart or /reset, so a stale high cursor would leave us fetching
    forever. Re-probe from -1: only rebuild if the highest id now available is
    BELOW our cursor (evidence the stream restarted). A call that merely paused
    or ended returns the same ids -> keep the history we have, no flicker.
    """

    probe = get_windows(since=-1)

    if not probe:
        return  # still nothing (backend down or truly empty) -- retry next tick

    max_id = max(w["window_id"] for w in probe)

    if max_id < st.session_state.last_window_id:
        st.session_state.windows = []
        st.session_state.risk_engine = RiskEngine()
        st.session_state.last_window_id = -1
        process_windows(probe)

    st.session_state.empty_streak = 0


def poll_backend():
    """One poll cycle: fetch, feed, and handle the empty/restart case."""

    new_windows = get_windows()

    if not st.session_state.backend_connected:
        return  # down: keep last known values, retry next tick

    if new_windows:
        process_windows(new_windows)
        st.session_state.empty_streak = 0
        return

    # Connected but nothing new. Only meaningful once a call is under way.
    if st.session_state.last_window_id >= 0:
        st.session_state.empty_streak += 1
        if st.session_state.empty_streak >= EMPTY_RESET_TICKS:
            _rebuild_if_stream_restarted()


# =========================================================
# LIVEKIT
# =========================================================

def get_livekit_credentials():
    """
    Mint a LiveKit token for the operator's embedded monitor view.

    The dashboard is a WATCHER: publish=False, subscribe=True. It must never
    publish the operator's microphone into the room -- if it did, the scoring
    agent would score the operator's own voice alongside the caller's, on stage.

    make_token is imported lazily: server/tokens.py raises SystemExit at import
    time when LiveKit creds are missing, so a lazy import keeps that failure
    contained to the call panel (caught in render_livekit) instead of taking
    down the whole dashboard.
    """

    if not LIVEKIT_URL:
        raise RuntimeError(
            "LIVEKIT_URL is not set in the environment."
        )

    from server.tokens import make_token

    token = make_token(
        identity=f"voiceguard-monitor-{SESSION_ID}",
        room="demo",
        publish=False,
        subscribe=True,
    )

    return {
        "url": LIVEKIT_URL,
        "token": token,
    }


def render_livekit():
    """
    Render the LiveKit room inside Streamlit, ONCE per full script run.

    This lives OUTSIDE the auto-refresh fragment on purpose: re-executing
    components.html every second would tear down and rebuild the iframe, drop
    the WebRTC connection, and thrash the call. Only the operator's scoped JWT
    and the LiveKit URL cross into the browser -- never the API key/secret.
    """

    try:
        livekit_data = get_livekit_credentials()

        livekit_url = livekit_data["url"]
        token = livekit_data["token"]

    except Exception as exc:
        st.error(
            f"LiveKit connection setup failed: {exc}"
        )
        return

    html_path = os.path.join(
        os.path.dirname(__file__),
        "livekit",
        "index.html",
    )

    app_js_path = os.path.join(
        os.path.dirname(__file__),
        "livekit",
        "app.js",
    )

    if not os.path.exists(html_path):
        st.error(f"LiveKit UI not found: {html_path}")
        return

    if not os.path.exists(app_js_path):
        st.error(f"LiveKit JavaScript not found: {app_js_path}")
        return

    with open(html_path, "r", encoding="utf-8") as file:
        html = file.read()

    with open(app_js_path, "r", encoding="utf-8") as file:
        app_js = file.read()

    # ---------------------------------------------------------
    # Put credentials on window BEFORE app.js executes.
    # ---------------------------------------------------------

    credentials_script = f"""
    <script>
        window.VOICEGUARD_LIVEKIT_URL = {livekit_url!r};
        window.VOICEGUARD_LIVEKIT_TOKEN = {token!r};
    </script>
    """

    # Remove the external app.js reference from index.html.
    html = html.replace('<script src="./app.js"></script>', "")

    # Inject credentials + app.js immediately before </body>.
    html = html.replace(
        "</body>",
        credentials_script
        + "<script>"
        + app_js
        + "</script>"
        + "</body>",
    )

    components.html(html, height=500, scrolling=False)


# =========================================================
# LIVE REGION  (auto-refreshing fragment)
# =========================================================
#
# Everything that reflects backend data lives here and ONLY here. The fragment
# reruns every POLL_INTERVAL_S; nothing outside it (the LiveKit embed above)
# is touched, so the call stays connected while these values move.
# =========================================================

@st.fragment(run_every=POLL_INTERVAL_S)
def live_region():

    poll_backend()

    engine = st.session_state.risk_engine
    windows = st.session_state.windows

    # -----------------------------------------------------
    # STATUS ROW
    # -----------------------------------------------------

    status_left, status_middle, status_right = st.columns(3)

    with status_left:
        if st.session_state.backend_connected:
            st.success("● LIVE")
        else:
            st.warning("● DISCONNECTED — retrying")

    with status_middle:
        st.write(f"**Session:** `{SESSION_ID}`")

    with status_right:
        if windows:
            duration = windows[-1]["t_end"]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            st.write(f"**Call duration:** {minutes:02d}:{seconds:02d}")
        else:
            st.write("**Call duration:** —")

    st.divider()

    # Risk context toggles (operator inputs). Inside the fragment so toggling
    # reruns only this region -- the embed in the body must not reload. Their
    # values persist in session_state and are read by process_windows each poll.
    st.subheader("Risk Context")

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.checkbox("Unknown caller ID", key="unknown_caller")
    with cc2:
        st.checkbox("High-value transaction", key="high_value_txn")
    with cc3:
        st.checkbox("Outside business hours", key="off_hours")
    with cc4:
        st.checkbox("First contact", key="first_contact")

    st.divider()

    risk_state = engine.state()

    # -----------------------------------------------------
    # CURRENT RISK + RISK OVER TIME
    # -----------------------------------------------------

    left, right = st.columns([1, 2])

    with left:

        st.subheader("Current Risk")

        if risk_state["risk"] is None:
            st.metric("Impersonation Risk", "—")
            st.info(
                "Risk will appear after the first valid audio window "
                "is analysed."
            )
        else:
            risk = risk_state["risk"]
            band = risk_state["band"]

            st.metric("Impersonation Risk", f"{risk:.2f}")

            if band == "RED":
                st.error("🔴 RED RISK")
            elif band == "AMBER":
                st.warning("🟠 AMBER RISK")
            else:
                st.success("🟢 GREEN RISK")

            st.progress(min(max(float(risk), 0.0), 1.0))

    with right:

        st.subheader("Risk Over Time")

        history = engine.history

        if not history:
            st.info(
                "Risk history will appear once audio windows are received "
                "from the backend."
            )
        else:
            x = [item["t_end"] for item in history]
            y_risk = [item["risk"] for item in history]
            y_raw = [item["prob_fake"] for item in history]

            fig = go.Figure()

            # Raw prob_fake, faint, so the smoothing is visible against it.
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_raw,
                    mode="lines",
                    name="Raw prob_fake",
                    line=dict(color="#8899aa", width=1),
                    opacity=0.45,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_risk,
                    mode="lines+markers",
                    name="Risk (smoothed)",
                    line=dict(color="#e0245e", width=2),
                )
            )

            fig.add_hline(y=L1_AMBER, line_dash="dash", annotation_text="AMBER")
            fig.add_hline(y=L1_RED, line_dash="dash", annotation_text="RED")

            fig.update_layout(
                height=350,
                yaxis=dict(range=[0, 1], title="Risk"),
                xaxis_title="Time (seconds)",
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=20, r=20, t=20, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # DETECTION PIPELINE  (fixed in step 2)
    # -----------------------------------------------------

    st.divider()

    st.write("**Detection Pipeline**")

    level_counts = {0: 0, 1: 0, 2: 0}
    for window in windows:
        level = window.get("level_resolved")
        if level in level_counts:
            level_counts[level] += 1

    total_windows = sum(level_counts.values())

    if total_windows:
        l0_pct = level_counts[0] / total_windows
        l1_pct = level_counts[1] / total_windows
        l2_pct = level_counts[2] / total_windows

        st.progress(l0_pct, text=f"L0 — Screening  {l0_pct:.0%}")
        st.progress(l1_pct, text=f"L1 — Detection  {l1_pct:.0%}")
        st.progress(l2_pct, text=f"L2 — Attribution  {l2_pct:.0%}")
    else:
        st.info("Waiting for detection windows...")

    # Speaker verification (L2 / speaker_sim) is stubbed and not demoed --
    # speaker_sim is always None. The field is kept in the window data, just
    # not surfaced. No display here on purpose.

    # -----------------------------------------------------
    # LATEST WINDOW
    # -----------------------------------------------------

    if windows:

        st.divider()

        st.subheader("Latest Detection Window")

        latest = windows[-1]

        w1, w2, w3, w4 = st.columns(4)

        with w1:
            st.metric("Window", latest["window_id"])
        with w2:
            st.metric("Fake Probability", f"{latest['prob_fake']:.2f}")
        with w3:
            st.metric("Speech Ratio", f"{latest['speech_ratio']:.0%}")
        with w4:
            st.metric("Latency", f"{latest['latency_ms']} ms")

    else:

        st.info("Waiting for audio analysis from the detection backend.")

    # -----------------------------------------------------
    # RED ALERT
    # -----------------------------------------------------

    if risk_state["band"] == "RED":

        st.divider()

        st.error(
            "⚠️ HIGH IMPERSONATION RISK — the current risk level has reached "
            "RED. Secondary verification is recommended before approving a "
            "sensitive action."
        )

        a1, a2, a3 = st.columns(3)
        with a1:
            st.button("📞 Request Call-back")
        with a2:
            st.button("🚨 Escalate")
        with a3:
            st.button("Dismiss")

    # -----------------------------------------------------
    # EVENT LOG
    # -----------------------------------------------------

    st.divider()

    st.subheader("Event Log")

    events = engine.events

    if not events:
        st.info("No risk events yet.")
    else:
        for event in reversed(events):
            st.write(f"• {event}")


# =========================================================
# PAGE BODY  (runs once per full script run)
# =========================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("🛡️ VoiceGuard")

with header_right:
    st.caption(f"Monitoring `{SESSION_ID}` · refresh {POLL_INTERVAL_S:.0f}s")

st.divider()

# LiveKit embed: rendered ONCE, outside the fragment, so the call survives the
# 1 s refresh loop below.
render_livekit()

st.divider()

# The live, auto-refreshing region. ALL interactive widgets live inside it, so
# that using them reruns only the fragment -- never the body, which would reload
# the embed above and drop the call.
live_region()
