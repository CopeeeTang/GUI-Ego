"""
Proactive Intervention Evaluation Metrics (RQ1)

Three-layer evaluation:
  Layer 1 — Trigger Timing: Did the system trigger at the right time?
  Layer 2 — Content Quality: Was the intervention content helpful?
  Layer 3 — System Quality: Overall trigger behavior analysis

Metrics:
  - Trigger Precision / Recall / F1 (with soft window matching)
  - Timing MAE (Mean Absolute Error to nearest GT trigger)
  - Content Semantic Similarity (sentence-transformer cosine sim)
  - LLM-as-Judge scores (relevance, helpfulness)
  - False Positive classification (benign vs harmful)
  - Intervention Density (triggers per minute)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.proactive.generator import Intervention
from src.proactive.gt_generator import GroundTruthTrigger

logger = logging.getLogger(__name__)


@dataclass
class TriggerMatch:
    """A match between a predicted trigger and ground truth."""
    pred_trigger: Intervention
    gt_trigger: Optional[GroundTruthTrigger]
    time_error: float = 0.0     # seconds
    is_true_positive: bool = False
    is_false_positive: bool = False
    is_benign_fp: bool = False   # FP but still helpful
    content_similarity: float = 0.0
    llm_judge_score: Optional[dict] = None


@dataclass
class ProactiveMetrics:
    """Complete evaluation results for proactive intervention."""
    # Layer 1: Trigger Timing
    trigger_precision: float = 0.0
    trigger_recall: float = 0.0
    trigger_f1: float = 0.0
    timing_mae: float = 0.0        # mean absolute error (seconds)
    timing_median_error: float = 0.0

    # Layer 2: Content Quality (TP triggers only)
    avg_semantic_similarity: float = 0.0
    avg_llm_relevance: float = 0.0
    avg_llm_helpfulness: float = 0.0

    # Layer 3: System Quality
    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    benign_fp: int = 0
    harmful_fp: int = 0
    triggers_per_minute: float = 0.0
    avg_latency_ms: float = 0.0

    # Per-type breakdown
    per_type_f1: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "═══ Proactive Intervention Metrics ═══",
            f"  Trigger P/R/F1: {self.trigger_precision:.3f} / {self.trigger_recall:.3f} / {self.trigger_f1:.3f}",
            f"  Timing MAE: {self.timing_mae:.2f}s (median: {self.timing_median_error:.2f}s)",
            f"  TP: {self.true_positives}, FP: {self.false_positives} (benign: {self.benign_fp}, harmful: {self.harmful_fp}), FN: {self.false_negatives}",
            f"  Content Similarity: {self.avg_semantic_similarity:.3f}",
            f"  LLM Judge — Relevance: {self.avg_llm_relevance:.2f}, Helpfulness: {self.avg_llm_helpfulness:.2f}",
            f"  Triggers/min: {self.triggers_per_minute:.2f}, Avg latency: {self.avg_latency_ms:.0f}ms",
        ]
        if self.per_type_f1:
            lines.append("  Per-type F1:")
            for ttype, f1 in sorted(self.per_type_f1.items()):
                lines.append(f"    {ttype}: {f1:.3f}")
        return "\n".join(lines)


class ProactiveEvaluator:
    """Evaluate proactive intervention system against ground truth."""

    def __init__(
        self,
        soft_window_sec: float = 3.0,
        llm_judge=None,             # Optional VLM for content judging
    ):
        self.soft_window_sec = soft_window_sec
        self.llm_judge = llm_judge
        self._embed_model = None

    def evaluate(
        self,
        predictions: list[Intervention],
        ground_truth: list[GroundTruthTrigger],
        session_duration_sec: float,
    ) -> ProactiveMetrics:
        """
        Full evaluation of proactive predictions against ground truth.

        Uses soft matching: a prediction matches a GT trigger if it falls
        within the GT's valid_window or within ±soft_window_sec.
        """
        matches = self._match_triggers(predictions, ground_truth)

        # --- Layer 1: Trigger Timing ---
        tp = sum(1 for m in matches if m.is_true_positive)
        fp = sum(1 for m in matches if m.is_false_positive)
        fn = len(ground_truth) - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        # Timing errors for TPs
        tp_errors = [m.time_error for m in matches if m.is_true_positive]
        mae = np.mean(tp_errors) if tp_errors else 0.0
        median_err = np.median(tp_errors) if tp_errors else 0.0

        # --- Layer 2: Content Quality (TPs only) ---
        similarities = []
        relevance_scores = []
        helpfulness_scores = []

        for m in matches:
            if not m.is_true_positive or m.gt_trigger is None:
                continue

            # Semantic similarity
            sim = self._semantic_similarity(
                m.pred_trigger.content, m.gt_trigger.expected_content
            )
            m.content_similarity = sim
            similarities.append(sim)

            # LLM-as-Judge (if available)
            if self.llm_judge:
                scores = self._llm_judge_content(
                    m.pred_trigger.content,
                    m.gt_trigger.expected_content,
                    m.gt_trigger.trigger_type.value,
                )
                m.llm_judge_score = scores
                relevance_scores.append(scores.get("relevance", 0))
                helpfulness_scores.append(scores.get("helpfulness", 0))

        # --- Layer 3: System Quality ---
        # Classify FPs as benign or harmful
        benign_fp = 0
        harmful_fp = 0
        for m in matches:
            if m.is_false_positive:
                if self.llm_judge:
                    is_benign = self._judge_fp_benignness(m.pred_trigger)
                    m.is_benign_fp = is_benign
                    if is_benign:
                        benign_fp += 1
                    else:
                        harmful_fp += 1
                else:
                    harmful_fp += 1

        duration_min = session_duration_sec / 60.0
        triggers_per_min = len(predictions) / max(0.01, duration_min)
        avg_latency = np.mean([p.latency_ms for p in predictions]) if predictions else 0

        # Per-type F1
        per_type_f1 = self._per_type_f1(matches, ground_truth)

        return ProactiveMetrics(
            trigger_precision=precision,
            trigger_recall=recall,
            trigger_f1=f1,
            timing_mae=mae,
            timing_median_error=median_err,
            avg_semantic_similarity=np.mean(similarities) if similarities else 0,
            avg_llm_relevance=np.mean(relevance_scores) if relevance_scores else 0,
            avg_llm_helpfulness=np.mean(helpfulness_scores) if helpfulness_scores else 0,
            total_predictions=len(predictions),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            benign_fp=benign_fp,
            harmful_fp=harmful_fp,
            triggers_per_minute=triggers_per_min,
            avg_latency_ms=avg_latency,
            per_type_f1=per_type_f1,
        )

    def _match_triggers(
        self,
        predictions: list[Intervention],
        ground_truth: list[GroundTruthTrigger],
    ) -> list[TriggerMatch]:
        """Match predictions to GT using soft window, greedy nearest-first."""
        matches = []
        used_gt = set()

        # Sort predictions by time
        preds_sorted = sorted(predictions, key=lambda p: p.timestamp)

        for pred in preds_sorted:
            best_gt = None
            best_error = float("inf")

            for i, gt in enumerate(ground_truth):
                if i in used_gt:
                    continue

                # Check if prediction falls within GT's valid window
                w_start = gt.valid_window[0] if gt.valid_window[0] > 0 else gt.timestamp - self.soft_window_sec
                w_end = gt.valid_window[1] if gt.valid_window[1] > 0 else gt.timestamp + self.soft_window_sec

                if w_start <= pred.timestamp <= w_end:
                    error = abs(pred.timestamp - gt.timestamp)
                    if error < best_error:
                        best_error = error
                        best_gt = (i, gt)

            if best_gt is not None:
                idx, gt = best_gt
                used_gt.add(idx)
                matches.append(TriggerMatch(
                    pred_trigger=pred,
                    gt_trigger=gt,
                    time_error=best_error,
                    is_true_positive=True,
                ))
            else:
                matches.append(TriggerMatch(
                    pred_trigger=pred,
                    gt_trigger=None,
                    is_false_positive=True,
                ))

        return matches

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute sentence-level cosine similarity."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                # Fallback: word overlap
                words_a = set(text_a.lower().split())
                words_b = set(text_b.lower().split())
                if not words_a or not words_b:
                    return 0.0
                return len(words_a & words_b) / len(words_a | words_b)

        embs = self._embed_model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def _llm_judge_content(self, pred_content: str, gt_content: str,
                           trigger_type: str) -> dict:
        """Use LLM to judge content quality."""
        prompt = f"""Rate the quality of a proactive intervention in a cooking assistant.

Trigger type: {trigger_type}
Expected content: {gt_content}
Predicted content: {pred_content}

Rate on two dimensions (0-5 scale each):
1. Relevance: How relevant is the prediction to the expected intervention?
2. Helpfulness: How helpful would this intervention be to the cook?

Respond in JSON: {{"relevance": 0-5, "helpfulness": 0-5, "reasoning": "brief explanation"}}"""

        try:
            result = self.llm_judge.generate_json(prompt)
            return {
                "relevance": float(result.get("relevance", 0)),
                "helpfulness": float(result.get("helpfulness", 0)),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return {"relevance": 0, "helpfulness": 0}

    def _judge_fp_benignness(self, pred: Intervention) -> bool:
        """Determine if a false positive is benign (still helpful) or harmful."""
        prompt = f"""A cooking assistant generated this proactive intervention, but it wasn't in the ground truth:

Intervention: {pred.content}
Timestamp: {pred.timestamp:.1f}s

Is this intervention still potentially helpful to the cook, or is it noise/spam?
Respond in JSON: {{"is_benign": true/false, "reason": "brief explanation"}}"""

        try:
            result = self.llm_judge.generate_json(prompt)
            return result.get("is_benign", False)
        except Exception:
            return False

    def _per_type_f1(self, matches: list[TriggerMatch],
                     ground_truth: list[GroundTruthTrigger]) -> dict[str, float]:
        """Compute F1 per trigger type."""
        from collections import Counter
        gt_types = Counter(gt.trigger_type.value for gt in ground_truth)
        tp_types = Counter()
        fp_types = Counter()

        for m in matches:
            if m.is_true_positive and m.gt_trigger:
                tp_types[m.gt_trigger.trigger_type.value] += 1
            elif m.is_false_positive:
                fp_types[m.pred_trigger.trigger_type] += 1

        result = {}
        for ttype in gt_types:
            tp = tp_types.get(ttype, 0)
            fp = fp_types.get(ttype, 0)
            fn = gt_types[ttype] - tp
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            result[ttype] = 2 * p * r / max(1e-8, p + r)

        return result
