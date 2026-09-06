"""
VoiceGuard Risk Engine

Responsibilities:
    1. Receive real WindowScore data from the backend.
    2. Ignore windows with insufficient speech.
    3. Smooth raw deepfake probabilities using EMA.
    4. Apply contextual risk multipliers.
    5. Convert risk into GREEN / AMBER / RED.
    6. Apply hysteresis to prevent rapid de-escalation.
    7. Maintain risk history for the dashboard.
    8. Record meaningful risk events.

The RiskEngine does NOT:
    - run the ML model
    - modify prob_fake
    - create fake detection results
    - determine level_resolved
    - perform audio processing
"""

import sys
from datetime import datetime
from pathlib import Path

# The calibrated thresholds live in common/config.py -- measured on the real
# model, not guessed. Import them; never hardcode. At the old placeholder
# L1_AMBER = 0.45 a genuine human speaker (raw prob_fake peaks ~0.55 live)
# trips the alert on stage. Ensure repo root is importable here regardless of
# how the dashboard is launched: this module may be imported before
# dashboard.py has extended sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.config import (
    L1_AMBER,
    L1_RED,
    EMA_ALPHA,
    HYSTERESIS_CLEAN_WINDOWS,
    MIN_SPEECH_RATIO,
)

class RiskEngine:

    def __init__(self, alpha=EMA_ALPHA):

        # -------------------------------------------------
        # SMOOTHING
        # -------------------------------------------------

        self.alpha = alpha

        # No risk exists until the first valid backend
        # WindowScore is received.
        self.ema = None

        # -------------------------------------------------
        # CURRENT STATE
        # -------------------------------------------------

        self.band = "GREEN"

        # Number of consecutive clean windows.
        self.clean_streak = 0

        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        # Used by the dashboard to plot risk over time.
        self.history = []

        # Human-readable risk events.
        self.events = []


    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(self, window: dict, context: dict | None = None):
        """
        Process one real WindowScore from the backend.

        Expected window fields:

            session_id
            window_id
            t_start
            t_end
            prob_fake
            level_resolved
            speech_ratio
            latency_ms
            model_version

        context contains only contextual information supplied
        by the product/UI layer.

        Current supported context:

            high_value_txn
            off_hours
            first_contact

        Returns the current RiskEngine state.
        """

        if context is None:
            context = {}


        # -------------------------------------------------
        # 1. SPEECH FILTER
        # -------------------------------------------------

        speech_ratio = float(
            window["speech_ratio"]
        )

        if speech_ratio < MIN_SPEECH_RATIO:

            # Do not update EMA.
            # Do not change the risk band.
            # Do not add a risk-history point.
            #
            # The architecture says low-speech windows
            # should not influence the detector/risk.

            return self.state()


        # -------------------------------------------------
        # 2. RAW MODEL OUTPUT
        # -------------------------------------------------

        prob_fake = float(
            window["prob_fake"]
        )

        # Keep this value within the mathematically valid
        # probability range.
        #
        # This does NOT change the backend value stored
        # in the WindowScore. It only protects the risk
        # calculation from invalid input.

        if not 0.0 <= prob_fake <= 1.0:
            raise ValueError(
                "prob_fake must be between 0.0 and 1.0"
            )


        # -------------------------------------------------
        # 3. EMA SMOOTHING
        # -------------------------------------------------

        if self.ema is None:

            # First valid speech window.
            self.ema = prob_fake

        else:

            self.ema = (
                self.alpha * prob_fake
                + (1.0 - self.alpha) * self.ema
            )


        # -------------------------------------------------
        # 4. CONTEXTUAL RISK
        # -------------------------------------------------

        multiplier = self._context_multiplier(
            context
        )

        risk = min(
            1.0,
            self.ema * multiplier
        )


        # -------------------------------------------------
        # 5. DETERMINE NEW RISK BAND
        # -------------------------------------------------

        new_band = self._band_from_risk(
            risk
        )


        # -------------------------------------------------
        # 6. HYSTERESIS
        # -------------------------------------------------

        self._apply_hysteresis(
            new_band,
            risk
        )


        # -------------------------------------------------
        # 7. STORE HISTORY
        # -------------------------------------------------

        self.history.append(
            {
                "window_id": window["window_id"],
                "t_start": window["t_start"],
                "t_end": window["t_end"],

                # Raw backend output.
                "prob_fake": prob_fake,

                # Smoothed model signal.
                "ema": self.ema,

                # Final context-adjusted risk.
                "risk": risk,

                # Current displayed band.
                "band": self.band,
            }
        )


        # -------------------------------------------------
        # 8. RETURN CURRENT STATE
        # -------------------------------------------------

        return self.state()


    # =====================================================
    # CONTEXT MULTIPLIERS
    # =====================================================

    def _context_multiplier(self, context: dict) -> float:
        """
        Calculate the contextual risk multiplier.

        These factors are contextual inputs to the product;
        they are not ML predictions.

        Supported factors:

            high_value_txn
            off_hours
            first_contact
        """

        multiplier = 1.0


        # High-value transaction
        if context.get("high_value_txn", False):

            multiplier *= 1.35


        # Outside normal business hours
        if context.get("off_hours", False):

            multiplier *= 1.10


        # First contact
        if context.get("first_contact", False):

            multiplier *= 1.15


        return multiplier


    # =====================================================
    # RISK BAND
    # =====================================================

    @staticmethod
    def _band_from_risk(risk: float) -> str:
        """
        Convert the calculated risk into a band.

            GREEN: risk < L1_AMBER
            AMBER: L1_AMBER <= risk < L1_RED
            RED:   risk >= L1_RED
        """

        if risk >= L1_RED:

            return "RED"

        if risk >= L1_AMBER:

            return "AMBER"

        return "GREEN"


    # =====================================================
    # HYSTERESIS
    # =====================================================

    def _apply_hysteresis(
        self,
        new_band: str,
        risk: float,
    ):
        """
        Escalation is immediate.

        De-escalation requires consecutive clean windows.

        Example:

            GREEN → AMBER    immediate
            AMBER → RED      immediate

            RED → AMBER      does NOT immediately happen
            RED → GREEN      requires clean streak

        This prevents the displayed risk from rapidly
        oscillating during a live call.
        """

        current_rank = self._rank(
            self.band
        )

        new_rank = self._rank(
            new_band
        )


        # -------------------------------------------------
        # ESCALATION
        # -------------------------------------------------

        if new_rank > current_rank:

            self._log(
                f"Risk crossed {new_band} "
                f"({risk:.2f})"
            )

            self.band = new_band

            self.clean_streak = 0

            return


        # -------------------------------------------------
        # CLEAN WINDOW
        # -------------------------------------------------

        if new_band == "GREEN":

            self.clean_streak += 1

            if (
                self.clean_streak
                >= HYSTERESIS_CLEAN_WINDOWS
            ):

                if self.band != "GREEN":

                    self._log(
                        "Risk returned to GREEN"
                    )

                self.band = "GREEN"

                self.clean_streak = 0

            return


        # -------------------------------------------------
        # STILL ELEVATED
        # -------------------------------------------------

        # If the calculated band is AMBER or RED but
        # doesn't represent an escalation, don't count
        # this as a clean window.

        self.clean_streak = 0


    # =====================================================
    # BAND RANK
    # =====================================================

    @staticmethod
    def _rank(band: str) -> int:

        return {
            "GREEN": 0,
            "AMBER": 1,
            "RED": 2,
        }[band]


    # =====================================================
    # EVENT LOG
    # =====================================================

    def _log(self, message: str):

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        self.events.append(
            f"{timestamp}  {message}"
        )


    # =====================================================
    # CURRENT STATE
    # =====================================================

    def state(self) -> dict:
        """
        Return the current user-facing state.

        Before the first valid WindowScore:

            risk = None
            ema = None

        This is intentional. The system must not invent
        a starting risk value.
        """

        current_risk = None

        if self.history:

            current_risk = self.history[-1]["risk"]


        return {
            "risk": current_risk,
            "band": self.band,
            "ema": self.ema,
            "clean_streak": self.clean_streak,
            "events": self.events,
        }