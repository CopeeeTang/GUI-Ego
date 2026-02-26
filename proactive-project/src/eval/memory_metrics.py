"""
Memory System Evaluation Metrics (RQ2)

Evaluates each memory layer independently + end-to-end QA:

Layer 1 — Task Memory:
  - Step Detection Accuracy
  - Entity Tracking F1
  - Progress MAE

Layer 2 — Event Memory:
  - Event Recall@K
  - Temporal Localization IoU
  - Retrieval MRR (Mean Reciprocal Rank)

Layer 3 — Visual Memory:
  - Frame Retrieval Precision@K

Cross-Layer — End-to-End QA:
  - QA Accuracy (LLM-as-Judge)
  - Response Latency
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.egtea_loader import ActionClip, CookingSession
from src.memory.manager import MemoryManager
from src.models.base import BaseVLM

logger = logging.getLogger(__name__)


@dataclass
class MemoryQuery:
    """A test query for memory evaluation."""
    question: str
    expected_answer: str
    query_type: str             # past_action, object_location, step_progress, temporal
    relevant_timestamp: float   # GT time of the relevant event
    relevant_entities: list[str] = field(default_factory=list)


@dataclass
class MemoryMetrics:
    """Complete evaluation results for the memory system."""
    # Layer 1: Task Memory
    step_detection_accuracy: float = 0.0
    entity_tracking_precision: float = 0.0
    entity_tracking_recall: float = 0.0
    entity_tracking_f1: float = 0.0
    progress_mae: float = 0.0          # absolute error in progress %

    # Layer 2: Event Memory
    event_recall_at_1: float = 0.0
    event_recall_at_3: float = 0.0
    event_recall_at_5: float = 0.0
    event_mrr: float = 0.0             # Mean Reciprocal Rank
    temporal_localization_iou: float = 0.0

    # Layer 3: Visual Memory
    frame_retrieval_precision_at_1: float = 0.0
    frame_retrieval_precision_at_3: float = 0.0

    # Cross-Layer: End-to-End QA
    qa_accuracy: float = 0.0
    qa_avg_score: float = 0.0          # LLM-as-Judge 0-5
    avg_response_latency_ms: float = 0.0

    # Per-query-type breakdown
    per_type_accuracy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "═══ Memory System Metrics ═══",
            f"  Layer 1 (Task):",
            f"    Step Detection Acc: {self.step_detection_accuracy:.3f}",
            f"    Entity F1: {self.entity_tracking_f1:.3f}",
            f"    Progress MAE: {self.progress_mae:.1f}%",
            f"  Layer 2 (Event):",
            f"    Recall@1/3/5: {self.event_recall_at_1:.3f}/{self.event_recall_at_3:.3f}/{self.event_recall_at_5:.3f}",
            f"    MRR: {self.event_mrr:.3f}",
            f"    Temporal IoU: {self.temporal_localization_iou:.3f}",
            f"  Layer 3 (Visual):",
            f"    Frame P@1: {self.frame_retrieval_precision_at_1:.3f}",
            f"    Frame P@3: {self.frame_retrieval_precision_at_3:.3f}",
            f"  End-to-End QA:",
            f"    Accuracy: {self.qa_accuracy:.3f}",
            f"    Avg Score: {self.qa_avg_score:.2f}/5",
            f"    Latency: {self.avg_response_latency_ms:.0f}ms",
        ]
        if self.per_type_accuracy:
            lines.append("  Per-type QA Accuracy:")
            for qtype, acc in sorted(self.per_type_accuracy.items()):
                lines.append(f"    {qtype}: {acc:.3f}")
        return "\n".join(lines)


class MemoryQueryGenerator:
    """Generate evaluation queries from session annotations."""

    def generate_queries(self, session: CookingSession,
                         num_queries: int = 10) -> list[MemoryQuery]:
        """Generate diverse memory test queries for a session."""
        queries = []
        actions = session.actions

        if len(actions) < 3:
            return queries

        # 1. Past Action Queries — "What did you do before X?"
        for i in range(1, min(len(actions), num_queries // 4 + 1)):
            prev = actions[i-1]
            nouns_str = " and ".join(prev.nouns) if prev.nouns else "items"
            queries.append(MemoryQuery(
                question=f"What cooking action was performed right before '{actions[i].action_label}'?",
                expected_answer=f"{prev.action_label} — the cook was {prev.verb.lower()}ing the {nouns_str}",
                query_type="past_action",
                relevant_timestamp=prev.start_sec,
                relevant_entities=list(prev.nouns),
            ))

        # 2. Object Location Queries — "When was X last used?"
        entity_last_seen = {}
        for action in actions:
            for noun in action.nouns:
                entity_last_seen[noun] = action
        for noun, action in list(entity_last_seen.items())[:num_queries // 4]:
            queries.append(MemoryQuery(
                question=f"When was the {noun} last used in the cooking process?",
                expected_answer=f"The {noun} was last used during '{action.action_label}' at around {action.start_sec:.0f} seconds",
                query_type="object_location",
                relevant_timestamp=action.start_sec,
                relevant_entities=[noun],
            ))

        # 3. Step Progress Queries — "How many steps completed?"
        mid = len(actions) // 2
        unique_actions = len(set(a.action_label for a in actions))
        queries.append(MemoryQuery(
            question="How many cooking steps have been completed so far?",
            expected_answer=f"About {unique_actions} unique steps have been completed in this {session.recipe} recipe",
            query_type="step_progress",
            relevant_timestamp=actions[mid].end_sec,
        ))

        # 4. Temporal Queries — "What happened at time X?"
        for i in range(0, len(actions), max(1, len(actions) // (num_queries // 4))):
            if len(queries) >= num_queries:
                break
            action = actions[i]
            nouns_str = " and ".join(action.nouns) if action.nouns else "items"
            queries.append(MemoryQuery(
                question=f"What was the cook doing at {action.start_sec:.0f} seconds into the session?",
                expected_answer=f"{action.action_label} — {action.verb}ing the {nouns_str}",
                query_type="temporal",
                relevant_timestamp=action.start_sec,
                relevant_entities=list(action.nouns),
            ))

        return queries[:num_queries]


class MemoryEvaluator:
    """Evaluate the three-layer memory system."""

    def __init__(self, llm_judge: Optional[BaseVLM] = None):
        self.llm_judge = llm_judge

    def evaluate_task_memory(
        self,
        memory: MemoryManager,
        session: CookingSession,
    ) -> dict:
        """Evaluate Layer 1: Task Memory accuracy."""
        state = memory.task.get_state()
        gt_actions = session.actions

        # Step Detection: check which GT actions were correctly detected
        detected = set()
        for step in state.steps:
            if step.is_completed:
                detected.add(step.action)
        gt_action_set = set(a.action_label for a in gt_actions)

        correct = len(detected & gt_action_set)
        step_acc = correct / max(1, len(gt_action_set))

        # Entity Tracking: compare tracked entities vs GT
        gt_entities = set()
        for a in gt_actions:
            gt_entities.update(a.nouns)

        tracked = state.active_entities
        tp = len(tracked & gt_entities)
        entity_p = tp / max(1, len(tracked))
        entity_r = tp / max(1, len(gt_entities))
        entity_f1 = 2 * entity_p * entity_r / max(1e-8, entity_p + entity_r)

        # Progress accuracy
        gt_progress = 100.0  # at end of session, should be ~100%
        progress_mae = abs(state.progress_pct - gt_progress)

        return {
            "step_detection_accuracy": step_acc,
            "entity_tracking_precision": entity_p,
            "entity_tracking_recall": entity_r,
            "entity_tracking_f1": entity_f1,
            "progress_mae": progress_mae,
        }

    def evaluate_event_memory(
        self,
        memory: MemoryManager,
        queries: list[MemoryQuery],
    ) -> dict:
        """Evaluate Layer 2: Event Memory retrieval quality."""
        recall_at = {1: [], 3: [], 5: []}
        mrr_scores = []
        iou_scores = []

        for query in queries:
            # Retrieve events
            events = memory.events.retrieve(query.question, top_k=5)

            # Check if relevant entities appear in retrieved events
            relevant_found = False
            reciprocal_rank = 0.0

            for rank, event in enumerate(events, 1):
                # Match by entity overlap, action_label, or description keywords
                action_label = event.metadata.get("action_label", "")
                expected_lower = query.expected_answer.lower().replace("-", " ")
                is_relevant = (
                    any(e.lower() in [ent.lower() for ent in event.entities]
                        for e in query.relevant_entities) or
                    action_label.lower() in expected_lower or
                    any(noun.lower() in expected_lower
                        for noun in event.metadata.get("nouns", []))
                )
                if is_relevant and not relevant_found:
                    relevant_found = True
                    reciprocal_rank = 1.0 / rank

                    # Temporal IoU
                    if query.relevant_timestamp > 0:
                        pred_window = (event.timestamp - 2, event.timestamp + 2)
                        gt_window = (query.relevant_timestamp - 2, query.relevant_timestamp + 2)
                        intersection = max(0, min(pred_window[1], gt_window[1]) - max(pred_window[0], gt_window[0]))
                        union = max(pred_window[1], gt_window[1]) - min(pred_window[0], gt_window[0])
                        iou_scores.append(intersection / max(1e-8, union))

                # Recall@K
                for k in [1, 3, 5]:
                    if rank <= k and is_relevant:
                        recall_at[k].append(1.0)

            # If not found at any rank
            for k in [1, 3, 5]:
                if len(recall_at[k]) < len(mrr_scores) + 1:
                    recall_at[k].append(0.0)

            mrr_scores.append(reciprocal_rank)

        return {
            "event_recall_at_1": np.mean(recall_at[1]) if recall_at[1] else 0,
            "event_recall_at_3": np.mean(recall_at[3]) if recall_at[3] else 0,
            "event_recall_at_5": np.mean(recall_at[5]) if recall_at[5] else 0,
            "event_mrr": np.mean(mrr_scores) if mrr_scores else 0,
            "temporal_localization_iou": np.mean(iou_scores) if iou_scores else 0,
        }

    def evaluate_visual_memory(
        self,
        memory: MemoryManager,
        queries: list[MemoryQuery],
    ) -> dict:
        """Evaluate Layer 3: Visual Memory frame retrieval."""
        p_at_1 = []
        p_at_3 = []

        for query in queries:
            if query.relevant_timestamp <= 0:
                continue

            # Try to retrieve frame at relevant timestamp
            frame = memory.visual.get_frame_at_time(query.relevant_timestamp, tolerance_sec=3.0)

            if frame is not None:
                time_error = abs(frame.timestamp - query.relevant_timestamp)
                p_at_1.append(1.0 if time_error <= 3.0 else 0.0)

                # Check if action label matches
                if frame.action_label and query.expected_answer.lower() in frame.action_label.lower():
                    p_at_3.append(1.0)
                else:
                    p_at_3.append(0.0)
            else:
                p_at_1.append(0.0)
                p_at_3.append(0.0)

        return {
            "frame_retrieval_precision_at_1": np.mean(p_at_1) if p_at_1 else 0,
            "frame_retrieval_precision_at_3": np.mean(p_at_3) if p_at_3 else 0,
        }

    def evaluate_qa(
        self,
        memory: MemoryManager,
        vlm: BaseVLM,
        queries: list[MemoryQuery],
    ) -> dict:
        """Evaluate cross-layer end-to-end QA performance."""
        import time

        accuracies = []
        scores = []
        latencies = []
        per_type = {}

        for query in queries:
            t0 = time.time()
            answer = memory.query_with_vlm(vlm, query.question)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            # Judge answer quality
            if self.llm_judge:
                score_data = self._judge_answer(
                    query.question, query.expected_answer, answer
                )
                acc = 1.0 if score_data.get("correct", False) else 0.0
                score = score_data.get("score", 0)
            else:
                # Fallback: keyword overlap
                overlap = len(
                    set(query.expected_answer.lower().split()) &
                    set(answer.lower().split())
                )
                acc = 1.0 if overlap >= 2 else 0.0
                score = min(5, overlap)

            accuracies.append(acc)
            scores.append(score)

            # Per-type tracking
            if query.query_type not in per_type:
                per_type[query.query_type] = []
            per_type[query.query_type].append(acc)

        return {
            "qa_accuracy": np.mean(accuracies) if accuracies else 0,
            "qa_avg_score": np.mean(scores) if scores else 0,
            "avg_response_latency_ms": np.mean(latencies) if latencies else 0,
            "per_type_accuracy": {k: np.mean(v) for k, v in per_type.items()},
        }

    def evaluate_full(
        self,
        memory: MemoryManager,
        session: CookingSession,
        queries: list[MemoryQuery],
        vlm: Optional[BaseVLM] = None,
    ) -> MemoryMetrics:
        """Run full evaluation across all layers."""
        metrics = MemoryMetrics()

        # Layer 1
        task_results = self.evaluate_task_memory(memory, session)
        metrics.step_detection_accuracy = task_results["step_detection_accuracy"]
        metrics.entity_tracking_precision = task_results["entity_tracking_precision"]
        metrics.entity_tracking_recall = task_results["entity_tracking_recall"]
        metrics.entity_tracking_f1 = task_results["entity_tracking_f1"]
        metrics.progress_mae = task_results["progress_mae"]

        # Layer 2
        event_results = self.evaluate_event_memory(memory, queries)
        metrics.event_recall_at_1 = event_results["event_recall_at_1"]
        metrics.event_recall_at_3 = event_results["event_recall_at_3"]
        metrics.event_recall_at_5 = event_results["event_recall_at_5"]
        metrics.event_mrr = event_results["event_mrr"]
        metrics.temporal_localization_iou = event_results["temporal_localization_iou"]

        # Layer 3
        visual_results = self.evaluate_visual_memory(memory, queries)
        metrics.frame_retrieval_precision_at_1 = visual_results["frame_retrieval_precision_at_1"]
        metrics.frame_retrieval_precision_at_3 = visual_results["frame_retrieval_precision_at_3"]

        # Cross-Layer QA (requires VLM)
        if vlm:
            qa_results = self.evaluate_qa(memory, vlm, queries)
            metrics.qa_accuracy = qa_results["qa_accuracy"]
            metrics.qa_avg_score = qa_results["qa_avg_score"]
            metrics.avg_response_latency_ms = qa_results["avg_response_latency_ms"]
            metrics.per_type_accuracy = qa_results["per_type_accuracy"]

        return metrics

    def _judge_answer(self, question: str, expected: str, predicted: str) -> dict:
        """Use LLM to judge answer correctness with lenient semantic matching."""
        prompt = f"""You are evaluating a cooking assistant's memory-based QA response.
Judge whether the predicted answer correctly addresses the question.
Be LENIENT — partial matches, paraphrases, and semantically equivalent answers should count.

Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}

Scoring guide:
- correct: true if the predicted answer captures the KEY INFORMATION from the expected answer
  (e.g., if expected is "Cut-Tomato" and predicted mentions "cutting tomatoes", that's correct)
  (e.g., if expected says "at 45.2s" and predicted says "around 45 seconds", that's correct)
- score: 0=completely wrong/irrelevant, 1=vaguely related, 2=partially correct, 3=mostly correct with minor gaps, 4=correct with minor wording differences, 5=perfect match

Respond ONLY with JSON: {{"correct": true/false, "score": 0-5, "reason": "brief"}}"""

        try:
            result = self.llm_judge.generate_json(prompt)
            # Also do fallback keyword check: if predicted contains key terms from expected
            if not result.get("correct", False):
                expected_lower = expected.lower()
                predicted_lower = predicted.lower()
                # Check if key action label or entity appears in prediction
                key_terms = [w for w in expected_lower.replace("-", " ").split()
                             if len(w) > 2 and w not in {"the", "was", "and", "for", "with", "at", "during"}]
                match_count = sum(1 for t in key_terms if t in predicted_lower)
                if match_count >= max(1, len(key_terms) * 0.4):
                    result["correct"] = True
                    result["score"] = max(result.get("score", 0), 3)
            return result
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            # Fallback: keyword overlap check
            expected_lower = expected.lower()
            predicted_lower = predicted.lower()
            key_terms = [w for w in expected_lower.replace("-", " ").split()
                         if len(w) > 2 and w not in {"the", "was", "and", "for", "with", "at", "during"}]
            match_count = sum(1 for t in key_terms if t in predicted_lower)
            if match_count >= max(1, len(key_terms) * 0.4):
                return {"correct": True, "score": 3}
            return {"correct": False, "score": 0}
