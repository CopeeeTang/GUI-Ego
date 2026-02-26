"""Signal analyzer — detect fixation, HR/EDA spikes."""

import logging
import math
from typing import Optional

from .reader import SignalReader

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """Analyze physiological signals for anomalies at a given timestamp.

    Detects:
        - Gaze fixation: sustained gaze on a point (same fixation_id > threshold)
        - HR spike: heart rate > N standard deviations from running mean
        - EDA spike: electrodermal activity > N standard deviations from running mean
    """

    def __init__(
        self,
        reader: SignalReader,
        gaze_fixation_threshold_ms: float = 500.0,
        hr_spike_std_multiplier: float = 2.0,
        eda_spike_std_multiplier: float = 2.0,
    ):
        self.reader = reader
        self.gaze_fixation_threshold_ms = gaze_fixation_threshold_ms
        self.hr_spike_std = hr_spike_std_multiplier
        self.eda_spike_std = eda_spike_std_multiplier

        # Precompute HR and EDA statistics for spike detection
        self._hr_mean, self._hr_std = self._compute_stats(
            [h["hr"] for h in reader.hr_data]
        )
        self._eda_mean, self._eda_std = self._compute_stats(
            [e["eda"] for e in reader.eda_data]
        )

    def analyze_at(self, timestamp: float) -> dict:
        """Analyze all signals at the given timestamp.

        Args:
            timestamp: Absolute timestamp in seconds.

        Returns:
            Dict with keys: gaze_fixation, hr_spike, eda_spike (all bool),
            plus raw values when available.
        """
        result = {
            "gaze_fixation": False,
            "hr_spike": False,
            "eda_spike": False,
        }

        # Gaze fixation detection
        gaze_fixation, fixation_duration = self._detect_gaze_fixation(timestamp)
        result["gaze_fixation"] = gaze_fixation
        if fixation_duration is not None:
            result["fixation_duration_ms"] = fixation_duration

        # HR spike detection
        hr_samples = self.reader.get_hr_at(timestamp)
        if hr_samples:
            hr_val = hr_samples[0]["hr"]
            result["hr_value"] = hr_val
            if self._hr_std > 0:
                result["hr_spike"] = (
                    abs(hr_val - self._hr_mean) > self.hr_spike_std * self._hr_std
                )

        # EDA spike detection
        eda_samples = self.reader.get_eda_at(timestamp)
        if eda_samples:
            eda_val = eda_samples[0]["eda"]
            result["eda_value"] = eda_val
            if self._eda_std > 0:
                result["eda_spike"] = (
                    abs(eda_val - self._eda_mean) > self.eda_spike_std * self._eda_std
                )

        return result

    def _detect_gaze_fixation(self, timestamp: float) -> tuple[bool, Optional[float]]:
        """Detect if the user has a sustained gaze fixation at this timestamp.

        A fixation is detected when consecutive gaze samples share the same
        fixation_id for longer than the threshold duration.

        Returns:
            Tuple of (is_fixation, duration_ms or None).
        """
        # Get a wider window of gaze data to detect fixation duration
        gaze = self.reader.get_gaze_at(timestamp, window_sec=1.0)
        if not gaze:
            return False, None

        # Find the fixation_id at the target timestamp
        closest = min(gaze, key=lambda g: abs(g["time_s"] - timestamp))
        fixation_id = closest.get("fixation_id")

        if fixation_id is None:
            return False, None

        # Find all consecutive samples with the same fixation_id
        matching = [g for g in gaze if g.get("fixation_id") == fixation_id]
        if len(matching) < 2:
            return False, None

        matching.sort(key=lambda g: g["time_s"])
        duration_s = matching[-1]["time_s"] - matching[0]["time_s"]
        duration_ms = duration_s * 1000.0

        is_fixation = duration_ms >= self.gaze_fixation_threshold_ms
        return is_fixation, duration_ms

    @staticmethod
    def _compute_stats(values: list[float]) -> tuple[float, float]:
        """Compute mean and standard deviation."""
        if not values:
            return 0.0, 0.0

        n = len(values)
        mean = sum(values) / n

        if n < 2:
            return mean, 0.0

        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        return mean, std
