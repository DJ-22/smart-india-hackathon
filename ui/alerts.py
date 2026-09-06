"""
VoiceGuard Alerting Layer

Responsibilities:
    - Display the appropriate alert based on the RiskEngine state.
    - Recommend secondary verification when risk is RED.
    - Provide response actions for the operator.
    - Keep sensitive approval actions disabled.

This module does NOT:
    - calculate risk
    - modify prob_fake
    - run the detection models
    - create detection results
    - determine GREEN / AMBER / RED
"""

import streamlit as st


# =========================================================
# ALERT DISPLAY
# =========================================================

def render_alert(risk_state: dict):
    """
    Render the appropriate alert for the current risk state.

    Expected risk_state:

        {
            "risk": float | None,
            "band": "GREEN" | "AMBER" | "RED",
            ...
        }

    No alert is shown if there is no valid risk yet.
    """

    risk = risk_state.get("risk")
    band = risk_state.get("band")

    # -----------------------------------------------------
    # NO DETECTION DATA
    # -----------------------------------------------------

    if risk is None:
        return None

    # -----------------------------------------------------
    # GREEN
    # -----------------------------------------------------

    if band == "GREEN":

        st.success(
            "🟢 No elevated impersonation risk detected."
        )

        return None

    # -----------------------------------------------------
    # AMBER
    # -----------------------------------------------------

    if band == "AMBER":

        st.warning(
            f"""
            🟠 **ELEVATED IMPERSONATION RISK**

            Current risk: **{risk:.2f}**

            Continue monitoring the call and consider
            secondary verification before approving a
            sensitive action.
            """
        )

        return None

    # -----------------------------------------------------
    # RED
    # -----------------------------------------------------

    if band == "RED":

        return render_red_alert(risk)


    return None


# =========================================================
# RED ALERT
# =========================================================

def render_red_alert(risk: float):
    """
    Render the RED-risk response interface.

    The buttons currently represent product actions.
    They do not claim to perform a real callback or
    escalation until those backend workflows exist.
    """

    st.error(
        f"""
        ⚠️ **HIGH IMPERSONATION RISK**

        Current risk: **{risk:.2f}**

        Voice analysis indicates a high level of
        potential impersonation risk.

        **Recommended action:** perform secondary
        verification before approving a sensitive action.
        """
    )

    st.markdown("### Recommended Response")

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # CALLBACK
    # -----------------------------------------------------

    with col1:

        callback = st.button(
            "📞 Request Call-back",
            key="alert_callback",
            use_container_width=True,
        )

        if callback:

            st.info(
                "Call-back verification requested."
            )

    # -----------------------------------------------------
    # ESCALATE
    # -----------------------------------------------------

    with col2:

        escalate = st.button(
            "🚨 Escalate",
            key="alert_escalate",
            use_container_width=True,
        )

        if escalate:

            st.warning(
                "Escalation requested."
            )

    # -----------------------------------------------------
    # DISMISS
    # -----------------------------------------------------

    with col3:

        dismiss = st.button(
            "Dismiss",
            key="alert_dismiss",
            use_container_width=True,
        )

        if dismiss:

            st.session_state["alert_dismissed"] = True

            st.rerun()


    # -----------------------------------------------------
    # SENSITIVE ACTION
    # -----------------------------------------------------

    st.markdown("### Sensitive Action")

    st.button(
        "Approve Transaction",
        disabled=True,
        use_container_width=True,
        help=(
            "Approval is disabled while the call is "
            "classified as high impersonation risk."
        ),
    )

    st.caption(
        "Secondary verification is required before "
        "approving a sensitive action."
    )