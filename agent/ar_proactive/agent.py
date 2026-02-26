"""Proactive AR Agent — RQ-driven Observe → Think → Act loop.

Supports two modes:
  1. Legacy sample mode: process sample directories with video/signals
  2. EGTEA mode: process EGTEA Gaze+ sessions with ground truth evaluation
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from agent.src.llm.factory import create_client
from .config import ARAgentConfig
from .context import StreamingContext

# Memory (RQ2)
from .memory.manager import MemoryManager
from .memory.types import FrameRecord, KeyEvent

# Task understanding (RQ1+RQ2)
from .task.knowledge import TaskKnowledgeExtractor
from .task.tracker import TaskTracker

# Video processing
from .video.change_detector import VisualChangeDetector
from .video.scene_analyzer import SceneAnalyzer

# Intervention (RQ1)
from .intervention.trigger import TriggerDecider
from .intervention.engine import InterventionEngine
from .intervention.types import Intervention

# Data loading
from .data.egtea_loader import EGTEALoader, EGTEASession

logger = logging.getLogger(__name__)


class ProactiveARAgent:
    """Orchestrator for the Proactive AR Smart Glasses Agent.

    New architecture:
      OBSERVE: frame extraction → visual change → scene analysis → signal analysis
      THINK:   FrameRecord → WorkingMemory → TaskTracker → StreamingContext → TriggerDecider
      ACT:     InterventionEngine → Intervention
    """

    def __init__(self, config: Optional[ARAgentConfig] = None):
        self.config = config or ARAgentConfig()

        # LLM client (shared across all components)
        self.llm_client = create_client(self.config.model_spec)

        # Scene analysis
        self.scene_analyzer = SceneAnalyzer(
            self.llm_client, max_tokens=self.config.max_tokens
        )

        # Task understanding
        self.task_extractor = TaskKnowledgeExtractor(
            self.llm_client, max_tokens=self.config.max_tokens
        )
        self.task_tracker = TaskTracker(
            self.llm_client,
            visual_change_threshold=self.config.step_visual_change_threshold,
        )

        # Intervention
        self.trigger_decider = TriggerDecider(
            cooldown_sec=self.config.cooldown_sec,
        )
        self.intervention_engine = InterventionEngine(
            llm_client=self.llm_client,
            min_confidence=self.config.min_confidence,
            max_tokens=self.config.max_tokens,
        )

        logger.info(
            f"ProactiveARAgent initialized with model={self.config.model_spec}"
        )

    # ── EGTEA Mode ────────────────────────────────────────────

    def process_egtea_session(
        self,
        session: EGTEASession,
        loader: EGTEALoader,
    ) -> dict:
        """Process a full EGTEA cooking session.

        Args:
            session: EGTEASession with clips, actions, gaze data.
            loader: EGTEALoader for frame iteration and gaze loading.

        Returns:
            Result dict with interventions, session_log, evaluation metrics.
        """
        logger.info(
            f"Processing EGTEA session: {session.name} "
            f"({session.recipe}, {session.num_actions} actions, "
            f"{len(session.clips)} clips)"
        )
        start_time = time.time()

        # ── Initialize components ────────────────────
        memory = MemoryManager(
            working_capacity=self.config.working_memory_capacity,
            max_events=self.config.max_key_events,
        )
        change_detector = VisualChangeDetector()

        # Load EGTEA gaze data for signal analysis
        gaze_data = loader.load_gaze(session)
        gaze_fps = 24  # EGTEA standard

        # ── Task Identification Phase ────────────────
        if self.config.use_egtea_gt_steps:
            # Use ground truth action sequence as task steps
            memory.persistent.set_task_from_egtea(session.recipe, session.actions)
            logger.info(f"Using EGTEA GT: {memory.persistent.total_steps} steps")
        else:
            # Identify task from initial frames via VLM
            initial_frames = self._collect_initial_frames(
                session, loader,
                n_frames=self.config.task_identification_frames,
            )
            if initial_frames:
                success = self.task_extractor.extract(
                    initial_frames, memory.persistent,
                    scene_context=f"This is a cooking session for: {session.recipe}",
                )
                if success:
                    logger.info(
                        f"Task identified: {memory.persistent.task_goal} "
                        f"({memory.persistent.total_steps} steps)"
                    )

        # ── Main Observe → Think → Act Loop ──────────
        interventions: list[Intervention] = []
        session_log: list[dict] = []
        frame_count = 0
        prev_action_label = ""

        for timestamp, frame, gt_action in loader.iter_session_frames(
            session,
            interval_sec=self.config.frame_interval_sec,
            max_clips=self.config.egtea_max_clips,
        ):
            frame_count += 1
            frame_log: dict = {
                "timestamp": timestamp,
                "frame_index": frame_count,
                "gt_action": gt_action.action_label if gt_action else None,
                "gt_verb": gt_action.verb if gt_action else None,
            }

            if self.config.verbose and frame_count % 5 == 1:
                gt_str = gt_action.action_label if gt_action else "?"
                logger.info(f"── Frame {frame_count} at {timestamp:.1f}s [{gt_str}] ──")

            # ── OBSERVE ──────────────────────────────
            frame_base64 = self._frame_to_base64(frame)
            visual_change = change_detector.compute_change(frame)

            # Scene analysis via VLM
            scene = self.scene_analyzer.analyze(frame_base64)

            # Gaze-based signal analysis (from EGTEA gaze data)
            signal_context = self._analyze_egtea_gaze(
                gaze_data, timestamp, session, gaze_fps
            )

            frame_log["visual_change"] = round(visual_change, 3)
            frame_log["scene_environment"] = scene.get("environment", "")
            frame_log["signals"] = signal_context

            # ── THINK ────────────────────────────────
            # Build FrameRecord
            record = FrameRecord(
                timestamp=timestamp,
                frame_base64=frame_base64,
                environment=scene.get("environment", ""),
                detected_objects=scene.get("objects", []),
                detected_activities=scene.get("activities", []),
                current_action=scene.get("current_action", ""),
                text_visible=scene.get("text_visible", []),
                potential_hazards=scene.get("potential_hazards", []),
                people_present=scene.get("people_present", False),
                visual_change_score=visual_change,
                signal_context=signal_context,
                scene_tags=scene.get("scene_tags", []),
                gt_action_label=gt_action.action_label if gt_action else "",
                gt_verb=gt_action.verb if gt_action else "",
                gt_nouns=gt_action.nouns if gt_action else [],
            )
            memory.add_frame(record)

            # Detect action boundary from GT (for evaluation logging)
            current_action_label = gt_action.action_label if gt_action else ""
            action_boundary = (
                current_action_label != prev_action_label
                and prev_action_label != ""
            )
            prev_action_label = current_action_label
            frame_log["gt_action_boundary"] = action_boundary

            # Task tracking (step detection)
            scene_desc = self._format_scene_text(scene)
            task_update = self.task_tracker.update(
                timestamp=timestamp,
                frame_base64=frame_base64,
                scene_description=scene_desc,
                visual_change=visual_change,
                memory=memory,
            )
            frame_log["task_update"] = task_update

            # Assemble StreamingContext
            ctx = StreamingContext.assemble(
                timestamp=timestamp,
                current_frame=record,
                persistent=memory.persistent,
                progress=memory.progress,
                working=memory.working,
                signal_context=signal_context,
                visual_change=visual_change,
                trigger_type="egtea_frame",
            )

            # Trigger evaluation (RQ1)
            trigger = self.trigger_decider.evaluate(ctx, task_update)
            frame_log["trigger"] = {
                "should_trigger": trigger.should_trigger,
                "mode": trigger.mode,
                "reasons": trigger.reasons,
            }

            # ── ACT ──────────────────────────────────
            intervention = None
            if trigger.should_trigger:
                intervention = self.intervention_engine.generate(ctx, trigger)

            if intervention:
                interventions.append(intervention)
                frame_log["intervention"] = intervention.to_dict()
                logger.info(
                    f"[{timestamp:.1f}s] INTERVENTION ({intervention.intervention_mode.value}): "
                    f"{intervention.content[:80]}"
                )
            else:
                frame_log["intervention"] = None

            session_log.append(frame_log)

        # ── Results ──────────────────────────────────
        elapsed = time.time() - start_time

        result = {
            "session": session.name,
            "recipe": session.recipe,
            "participant": session.participant,
            "gt_actions": session.num_actions,
            "frames_processed": frame_count,
            "interventions": [i.to_dict() for i in interventions],
            "intervention_count": len(interventions),
            "session_log": session_log,
            "memory_stats": memory.stats(),
            "processing_time_sec": round(elapsed, 2),
        }

        # Basic evaluation metrics
        result["eval"] = self._compute_eval_metrics(session_log, session)

        logger.info(
            f"Done: {frame_count} frames, {len(interventions)} interventions, "
            f"{elapsed:.1f}s elapsed"
        )

        return result

    def _collect_initial_frames(
        self,
        session: EGTEASession,
        loader: EGTEALoader,
        n_frames: int = 3,
    ) -> list[str]:
        """Collect the first N frames from a session for task identification."""
        frames = []
        for timestamp, frame, _ in loader.iter_session_frames(
            session, interval_sec=2.0, max_clips=3,
        ):
            frames.append(self._frame_to_base64(frame))
            if len(frames) >= n_frames:
                break
        return frames

    def _analyze_egtea_gaze(
        self,
        gaze_data: Optional[np.ndarray],
        timestamp: float,
        session: EGTEASession,
        fps: int = 24,
    ) -> dict:
        """Analyze EGTEA gaze data at a given timestamp.

        Converts EGTEA's frame-indexed gaze data to the agent's signal format.
        """
        if gaze_data is None:
            return {}

        # Convert timestamp to frame index relative to session start
        if session.actions:
            session_start_sec = session.actions[0].start_sec
        else:
            session_start_sec = 0.0

        frame_idx = int((timestamp - session_start_sec) * fps)
        if frame_idx < 0 or frame_idx >= len(gaze_data):
            return {}

        # Look at a window of frames for fixation detection
        window_frames = 12  # 0.5 sec at 24fps
        start_idx = max(0, frame_idx - window_frames)
        end_idx = min(len(gaze_data), frame_idx + 1)
        window = gaze_data[start_idx:end_idx]

        # Count fixation frames in window
        fixation_count = np.sum(window[:, 2] == 1)
        fixation_ratio = fixation_count / max(1, len(window))

        # Check for sustained fixation (most frames in window are fixation)
        is_fixation = fixation_ratio > 0.7
        fixation_duration_ms = int(fixation_count * (1000 / fps))

        # Current gaze position
        gaze_x = float(gaze_data[frame_idx, 0])
        gaze_y = float(gaze_data[frame_idx, 1])

        return {
            "gaze_fixation": is_fixation,
            "fixation_duration_ms": fixation_duration_ms if is_fixation else 0,
            "gaze_x": round(gaze_x, 3),
            "gaze_y": round(gaze_y, 3),
            "fixation_ratio": round(fixation_ratio, 3),
            # EGTEA doesn't have HR/EDA
            "hr_spike": False,
            "eda_spike": False,
        }

    def _compute_eval_metrics(self, session_log: list[dict], session: EGTEASession) -> dict:
        """Compute evaluation metrics from the session log.

        Metrics:
          - trigger_count: total triggers
          - trigger_at_boundary: triggers that coincide with GT action boundaries
          - boundary_coverage: fraction of GT boundaries that got a trigger within ±2 frames
          - intervention_types: distribution of intervention types
        """
        total_triggers = 0
        triggers_at_boundary = 0
        gt_boundaries = []
        trigger_timestamps = []

        for entry in session_log:
            trigger = entry.get("trigger", {})
            if trigger.get("should_trigger"):
                total_triggers += 1
                trigger_timestamps.append(entry["timestamp"])
                if entry.get("gt_action_boundary"):
                    triggers_at_boundary += 1

            if entry.get("gt_action_boundary"):
                gt_boundaries.append(entry["timestamp"])

        # Boundary coverage: fraction of GT boundaries with a nearby trigger
        covered = 0
        for boundary_ts in gt_boundaries:
            for trigger_ts in trigger_timestamps:
                if abs(trigger_ts - boundary_ts) <= 3.0:  # within 3 seconds
                    covered += 1
                    break

        boundary_coverage = covered / max(1, len(gt_boundaries))

        # Intervention type distribution
        type_counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        for entry in session_log:
            intv = entry.get("intervention")
            if intv:
                t = intv.get("intervention_type", "unknown")
                m = intv.get("intervention_mode", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
                mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            "total_triggers": total_triggers,
            "triggers_at_gt_boundary": triggers_at_boundary,
            "gt_boundary_count": len(gt_boundaries),
            "boundary_coverage": round(boundary_coverage, 3),
            "intervention_type_distribution": type_counts,
            "intervention_mode_distribution": mode_counts,
        }

    # ── Legacy Sample Mode ────────────────────────────────────

    def process_sample(self, sample_dir: str | Path) -> dict:
        """Process a single video sample (legacy mode).

        Preserved for backward compatibility with the old sample structure.
        """
        from .memory.store import TieredMemoryStore
        from .memory.importance import ImportanceScorer
        from .memory.retriever import MemoryRetriever
        from .memory.types import MemoryEntry, LongTermSummary
        from .video.frame_processor import FrameProcessor
        from .signals.reader import SignalReader
        from .signals.analyzer import SignalAnalyzer

        sample_dir = Path(sample_dir)
        video_path = sample_dir / "video" / "clip.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing sample (legacy mode): {sample_dir}")
        start_time = time.time()

        # Legacy components
        frame_processor = FrameProcessor(
            video_path,
            frame_interval_sec=self.config.frame_interval_sec,
            jpeg_quality=self.config.jpeg_quality,
        )
        change_detector = VisualChangeDetector()
        memory_store = TieredMemoryStore(
            short_term_capacity=self.config.short_term_capacity,
            mid_term_capacity=self.config.mid_term_capacity,
            importance_threshold=self.config.importance_threshold,
            long_term_summary_every=self.config.long_term_summary_every,
        )
        importance_scorer = ImportanceScorer(
            weight_visual_change=self.config.weight_visual_change,
            weight_semantic_novelty=self.config.weight_semantic_novelty,
            weight_signal_anomaly=self.config.weight_signal_anomaly,
            weight_scene_transition=self.config.weight_scene_transition,
        )

        signals_dir = sample_dir / "signals"
        signal_reader = SignalReader(signals_dir) if signals_dir.exists() else None
        signal_analyzer = None
        signal_time_offset = 0.0
        if signal_reader and signal_reader.gaze_data:
            signal_time_offset = signal_reader.time_range[0]
            signal_analyzer = SignalAnalyzer(
                signal_reader,
                gaze_fixation_threshold_ms=self.config.gaze_fixation_threshold_ms,
                hr_spike_std_multiplier=self.config.hr_spike_std_multiplier,
                eda_spike_std_multiplier=self.config.eda_spike_std_multiplier,
            )

        # Use new memory + trigger for THINK/ACT even in legacy mode
        memory = MemoryManager(
            working_capacity=self.config.working_memory_capacity,
        )

        interventions: list[Intervention] = []
        session_log: list[dict] = []
        frame_count = 0

        for timestamp, frame in frame_processor.iter_frames():
            frame_count += 1
            frame_log: dict = {"timestamp": timestamp, "frame_index": frame_count}

            # OBSERVE
            frame_base64 = frame_processor.frame_to_base64(frame)
            visual_change = change_detector.compute_change(frame)
            scene = self.scene_analyzer.analyze(frame_base64)

            signal_context: dict = {}
            if signal_analyzer:
                abs_time = signal_time_offset + timestamp
                signal_context = signal_analyzer.analyze_at(abs_time)

            frame_log["visual_change"] = round(visual_change, 3)
            frame_log["scene"] = scene
            frame_log["signals"] = signal_context

            # THINK (new system)
            record = FrameRecord(
                timestamp=timestamp,
                frame_base64=frame_base64,
                environment=scene.get("environment", ""),
                detected_objects=scene.get("objects", []),
                detected_activities=scene.get("activities", []),
                visual_change_score=visual_change,
                signal_context=signal_context,
                scene_tags=scene.get("scene_tags", []),
            )
            memory.add_frame(record)

            # Also store in legacy memory for backward compat
            importance = importance_scorer.score(
                visual_change=visual_change,
                detected_objects=scene.get("objects", []),
                detected_activities=scene.get("activities", []),
                recent_objects=memory_store.get_recent_objects(),
                recent_activities=memory_store.get_recent_activities(),
                signal_context=signal_context,
                scene_tags=scene.get("scene_tags", []),
            )
            legacy_entry = MemoryEntry(
                timestamp=timestamp, frame_base64=frame_base64,
                scene_description=scene.get("environment", ""),
                detected_objects=scene.get("objects", []),
                detected_activities=scene.get("activities", []),
                importance_score=importance,
                visual_change_score=visual_change,
                signal_context=signal_context,
                tags=scene.get("scene_tags", []),
            )
            memory_store.add(legacy_entry)
            frame_log["importance_score"] = round(importance, 3)

            # Task tracking
            task_update = self.task_tracker.update(
                timestamp=timestamp,
                frame_base64=frame_base64,
                scene_description=self._format_scene_text(scene),
                visual_change=visual_change,
                memory=memory,
            )

            # StreamingContext + Trigger
            ctx = StreamingContext.assemble(
                timestamp=timestamp, current_frame=record,
                persistent=memory.persistent, progress=memory.progress,
                working=memory.working, signal_context=signal_context,
                visual_change=visual_change,
            )
            trigger = self.trigger_decider.evaluate(ctx, task_update)

            # ACT
            intervention = None
            if trigger.should_trigger:
                intervention = self.intervention_engine.generate(ctx, trigger)

            if intervention:
                interventions.append(intervention)
                frame_log["intervention"] = intervention.to_dict()
            else:
                frame_log["intervention"] = None

            session_log.append(frame_log)

        frame_processor.close()
        elapsed = time.time() - start_time

        return {
            "sample_dir": str(sample_dir),
            "video_duration_sec": round(frame_processor.duration, 2),
            "frames_processed": frame_count,
            "interventions": [i.to_dict() for i in interventions],
            "intervention_count": len(interventions),
            "session_log": session_log,
            "memory_stats": memory.stats(),
            "processing_time_sec": round(elapsed, 2),
        }

    # ── Helpers ───────────────────────────────────────────────

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode a BGR frame as JPEG base64."""
        _, buffer = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        return base64.b64encode(buffer).decode("utf-8")

    def _format_scene_text(self, scene: dict) -> str:
        """Format scene analysis dict as text for prompts."""
        parts = [scene.get("environment", "unknown scene")]
        if scene.get("objects"):
            parts.append(f"Objects: {', '.join(scene['objects'][:8])}")
        if scene.get("activities"):
            parts.append(f"Activities: {', '.join(scene['activities'][:4])}")
        if scene.get("current_action"):
            parts.append(f"Current action: {scene['current_action']}")
        return ". ".join(parts)

    def save_results(self, results: dict | list, output_path: Optional[str | Path] = None) -> Path:
        """Save results to JSON file."""
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"ar_result_{timestamp_str}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cleaned = self._strip_base64(results)

        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def _strip_base64(self, data):
        """Recursively strip frame_base64 fields from data for output."""
        if isinstance(data, dict):
            return {
                k: self._strip_base64(v)
                for k, v in data.items()
                if k != "frame_base64"
            }
        elif isinstance(data, list):
            return [self._strip_base64(item) for item in data]
        return data
