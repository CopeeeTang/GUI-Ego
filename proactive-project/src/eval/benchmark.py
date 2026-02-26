"""
Benchmark Runner

End-to-end benchmark that orchestrates streaming simulation,
proactive detection, memory management, and evaluation.
"""

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from data.egtea_loader import EGTEALoader
from src.eval.memory_metrics import MemoryEvaluator, MemoryQueryGenerator
from src.eval.proactive_metrics import ProactiveEvaluator
from src.memory.event_memory import EventMemory
from src.memory.manager import MemoryManager
from src.memory.task_memory import TaskMemory
from src.memory.visual_memory import VisualMemory
from src.proactive.generator import ContentGenerator, Intervention, TemplateGenerator
from src.proactive.gt_generator import GroundTruthGenerator
from src.proactive.trigger import TriggerDecision, create_trigger
from src.streaming.stream_simulator import StreamSimulator

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run end-to-end evaluation for proactive + memory systems."""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.loader = EGTEALoader(
            root=self.config["dataset"]["root"],
            split=self.config["dataset"].get("split", 1),
        )
        self.simulator = StreamSimulator(
            loader=self.loader,
            sample_fps=self.config["streaming"]["sample_fps"],
        )
        gt_config = self.config.get("proactive", {}).get("gt", {})
        # Filter to only valid GroundTruthGenerator params
        valid_gt_keys = {"transition_gap_sec", "safety_lookahead_sec",
                         "idle_threshold_sec", "progress_interval", "soft_window_sec"}
        gt_config = {k: v for k, v in gt_config.items() if k in valid_gt_keys}
        self.gt_generator = GroundTruthGenerator(**gt_config)
        self.query_generator = MemoryQueryGenerator()

        # Output
        self.output_dir = Path(self.config["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_proactive_eval(
        self,
        vlm=None,
        trigger_method: str = "periodic",
        max_sessions: int = 5,
        recipe: Optional[str] = None,
    ) -> dict:
        """
        Run proactive intervention evaluation.

        Args:
            vlm: VLM model for trigger detection and content generation
            trigger_method: periodic | action_boundary | vlm_delta
            max_sessions: Number of sessions to evaluate
            recipe: Filter by recipe (e.g., "PastaSalad")
        """
        logger.info(f"Running proactive eval: method={trigger_method}, max_sessions={max_sessions}")

        # Setup
        trigger = create_trigger(trigger_method, vlm=vlm)
        generator = ContentGenerator(vlm) if vlm else TemplateGenerator()
        evaluator = ProactiveEvaluator(
            soft_window_sec=self.config["eval"]["proactive"]["soft_window_sec"],
            llm_judge=vlm if self.config["eval"]["proactive"].get("llm_judge_model") else None,
        )

        sessions = self.loader.iter_sessions(recipe=recipe)[:max_sessions]
        all_metrics = []

        for session in sessions:
            logger.info(f"  Evaluating session: {session.session_id}")

            # Generate ground truth
            gt_triggers = self.gt_generator.generate(session)
            if not gt_triggers:
                logger.warning(f"  No GT triggers for {session.session_id}, skipping")
                continue

            # Run streaming simulation
            predictions = []
            trigger.reset()
            recent_actions = []

            for frame, window in self.simulator.stream_session_windowed(session.session_id):
                decision = trigger.detect(frame, window)

                if decision.should_trigger:
                    # Generate content
                    if isinstance(generator, ContentGenerator):
                        intervention = generator.generate(
                            frame, decision.reason, recent_actions,
                            decision.method, decision.confidence,
                        )
                    else:
                        intervention = generator.generate(
                            frame, trigger_type=decision.reason or "default",
                        )
                    predictions.append(intervention)

                # Track actions
                if frame.current_action:
                    label = frame.current_action.action_label
                    if not recent_actions or recent_actions[-1] != label:
                        recent_actions.append(label)

            # Evaluate
            metrics = evaluator.evaluate(predictions, gt_triggers, session.duration_sec)
            all_metrics.append({
                "session_id": session.session_id,
                "recipe": session.recipe,
                "num_gt_triggers": len(gt_triggers),
                "num_predictions": len(predictions),
                "metrics": metrics.to_dict(),
            })
            logger.info(f"    F1={metrics.trigger_f1:.3f}, MAE={metrics.timing_mae:.2f}s")

        # Aggregate
        summary = self._aggregate_proactive_metrics(all_metrics)

        # Save results
        result = {
            "experiment": "proactive_eval",
            "trigger_method": trigger_method,
            "num_sessions": len(all_metrics),
            "aggregate": summary,
            "per_session": all_metrics,
        }
        self._save_results(result, f"proactive_{trigger_method}")
        return result

    def run_memory_eval(
        self,
        vlm=None,
        max_sessions: int = 5,
        queries_per_session: int = 10,
        recipe: Optional[str] = None,
    ) -> dict:
        """
        Run memory system evaluation.

        Streams each session through the memory manager, then
        evaluates memory quality at the end using generated queries.
        """
        logger.info(f"Running memory eval: max_sessions={max_sessions}")

        evaluator = MemoryEvaluator(llm_judge=vlm)
        sessions = self.loader.iter_sessions(recipe=recipe)[:max_sessions]
        all_metrics = []

        for session in sessions:
            logger.info(f"  Evaluating session: {session.session_id}")

            # Initialize memory
            memory = MemoryManager(
                task_memory=TaskMemory(),
                event_memory=EventMemory(
                    max_events=self.config["memory"]["event"].get("max_events", 200),
                    similarity_threshold=self.config["memory"]["event"].get("similarity_threshold", 0.3),
                ),
                visual_memory=VisualMemory(
                    recent_capacity=self.config["memory"]["visual"]["recent_frames"],
                    compressed_capacity=self.config["memory"]["visual"]["compressed_pool_size"],
                    compression_method=self.config["memory"]["visual"]["compression"],
                ),
            )
            memory.initialize(session)

            # Stream all frames through memory
            for frame in self.simulator.stream_session(session.session_id):
                memory.process_frame(
                    timestamp=frame.timestamp,
                    frame=frame.frame,
                    current_action=frame.current_action,
                )

            # Generate test queries
            queries = self.query_generator.generate_queries(
                session, num_queries=queries_per_session
            )

            # Evaluate
            metrics = evaluator.evaluate_full(
                memory, session, queries, vlm=vlm,
            )
            all_metrics.append({
                "session_id": session.session_id,
                "recipe": session.recipe,
                "num_queries": len(queries),
                "metrics": metrics.to_dict(),
            })
            logger.info(f"    Task Acc={metrics.step_detection_accuracy:.3f}, "
                        f"Event R@5={metrics.event_recall_at_5:.3f}, "
                        f"QA={metrics.qa_accuracy:.3f}")

        # Aggregate
        summary = self._aggregate_memory_metrics(all_metrics)

        result = {
            "experiment": "memory_eval",
            "num_sessions": len(all_metrics),
            "aggregate": summary,
            "per_session": all_metrics,
        }
        self._save_results(result, "memory")
        return result

    def _aggregate_proactive_metrics(self, results: list[dict]) -> dict:
        """Aggregate proactive metrics across sessions."""
        import numpy as np
        if not results:
            return {}
        keys = list(results[0]["metrics"].keys())
        agg = {}
        for key in keys:
            values = [r["metrics"][key] for r in results
                      if isinstance(r["metrics"].get(key), (int, float))]
            if values:
                agg[f"avg_{key}"] = float(np.mean(values))
                agg[f"std_{key}"] = float(np.std(values))
        return agg

    def _aggregate_memory_metrics(self, results: list[dict]) -> dict:
        """Aggregate memory metrics across sessions."""
        import numpy as np
        if not results:
            return {}
        keys = list(results[0]["metrics"].keys())
        agg = {}
        for key in keys:
            values = [r["metrics"][key] for r in results
                      if isinstance(r["metrics"].get(key), (int, float))]
            if values:
                agg[f"avg_{key}"] = float(np.mean(values))
        return agg

    def _save_results(self, result: dict, name: str):
        """Save evaluation results to JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
