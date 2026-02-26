"""Configuration for the Proactive AR Agent."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ARAgentConfig:
    """All tunable parameters for the Proactive AR Agent.

    Groups:
        LLM — model selection and token limits
        Video — frame extraction settings
        Memory — three-layer storage (RQ2)
        Task — task understanding (RQ1+RQ2)
        Signals — physiological anomaly detection
        Intervention — trigger and decision thresholds (RQ1)
        I/O — paths and logging

    Legacy fields (importance weights, tier capacities) are preserved
    for backward compatibility with old sample-based mode.
    """

    # ── LLM ──────────────────────────────────────
    model_spec: str = "claude:claude-sonnet-4-5"
    max_tokens: int = 1500

    # ── Video ────────────────────────────────────
    frame_interval_sec: float = 1.0
    jpeg_quality: int = 80

    # ── Memory (RQ2 three-layer) ─────────────────
    working_memory_capacity: int = 8
    max_key_events: int = 50
    compress_events_threshold: int = 30

    # ── Task (RQ1+RQ2) ──────────────────────────
    task_identification_frames: int = 3   # frames used for initial task ID
    step_detection_interval: int = 3      # check every N frames minimum
    step_visual_change_threshold: float = 0.15
    use_egtea_gt_steps: bool = False      # use EGTEA action labels as steps

    # ── Signals ──────────────────────────────────
    gaze_fixation_threshold_ms: float = 500.0
    hr_spike_std_multiplier: float = 2.0
    eda_spike_std_multiplier: float = 2.0

    # ── Intervention (RQ1) ───────────────────────
    min_confidence: float = 0.5
    cooldown_sec: float = 5.0

    # ── EGTEA ────────────────────────────────────
    egtea_data_root: str = "data/EGTEA_Gaze_Plus"
    egtea_session: Optional[str] = None   # e.g., "P01-R01-PastaSalad"
    egtea_max_clips: Optional[int] = None

    # ── I/O ──────────────────────────────────────
    data_root: str = "agent/example"
    output_dir: str = "output/ar_proactive"
    verbose: bool = False

    # ── Legacy (for old sample mode) ─────────────
    short_term_capacity: int = 6
    mid_term_capacity: int = 20
    importance_threshold: float = 0.4
    long_term_summary_every: int = 10
    weight_visual_change: float = 0.30
    weight_semantic_novelty: float = 0.30
    weight_signal_anomaly: float = 0.25
    weight_scene_transition: float = 0.15

    def __post_init__(self):
        """Validate legacy weight sum if legacy mode is used."""
        total = (
            self.weight_visual_change
            + self.weight_semantic_novelty
            + self.weight_signal_anomaly
            + self.weight_scene_transition
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Importance weights must sum to 1.0, got {total:.2f}"
            )
