"""
Ground Truth Generator for Proactive AR Intervention Evaluation.

Uses a strong VLM (Gemini-3-flash) as an expert annotator to generate
"ideal" intervention decisions at each action boundary in EGTEA sessions.

Features:
  - Single session or batch (all sessions) generation
  - Checkpoint/resume: saves after each boundary, resumes from last checkpoint
  - Rate limiting: configurable delay between API calls to avoid quota/disconnects
  - Robust error recovery: continues on failure, logs skipped boundaries

Usage:
  # Single session (2 boundaries for review)
  python3 -m agent.ar_proactive.eval.gt_generator \\
      --session P01-R01-PastaSalad --max-boundaries 2

  # Full session
  python3 -m agent.ar_proactive.eval.gt_generator --session P01-R01-PastaSalad

  # Batch: all sessions
  python3 -m agent.ar_proactive.eval.gt_generator --batch --delay 3

  # Resume interrupted batch
  python3 -m agent.ar_proactive.eval.gt_generator --batch --resume

Output: JSON with per-boundary annotation decisions.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Ground Truth Schema ──────────────────────────────────────────────

@dataclass
class GTAnnotation:
    """A single ground truth annotation at an action boundary."""

    # Context
    boundary_index: int
    timestamp_sec: float
    prev_action: str
    next_action: str
    recipe: str
    session: str

    # Annotation decision
    should_intervene: bool = False
    intervention_type: str = ""        # step_instruction, safety_warning, etc.
    intervention_mode: str = ""        # anticipatory, reactive
    content: str = ""                  # ideal intervention text
    priority: str = "medium"           # low, medium, high
    rationale: str = ""                # why this decision

    # Frame info
    frame_description: str = ""        # VLM description of the scene
    visible_hazards: list[str] = field(default_factory=list)
    visible_objects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "boundary_index": self.boundary_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "prev_action": self.prev_action,
            "next_action": self.next_action,
            "recipe": self.recipe,
            "session": self.session,
            "should_intervene": self.should_intervene,
            "intervention_type": self.intervention_type,
            "intervention_mode": self.intervention_mode,
            "content": self.content,
            "priority": self.priority,
            "rationale": self.rationale,
            "frame_description": self.frame_description,
            "visible_hazards": self.visible_hazards,
            "visible_objects": self.visible_objects,
        }


# ── Annotator Prompt ─────────────────────────────────────────────────

GT_SYSTEM_PROMPT = """\
You are an expert annotator for a Proactive AR Smart Glasses assistance system.

You are creating GROUND TRUTH labels for when and how the AR system should
proactively intervene to help a user performing a cooking task.

The user wears smart glasses with egocentric camera. The system can display
short text interventions (1-2 sentences) on the AR HUD.

You will see:
  - A frame from the user's egocentric camera at an action transition point
  - The action that just ended and the action about to begin
  - The recipe context

Your job: decide whether the system SHOULD intervene at this exact moment,
and if so, what it should say.

## Intervention Guidelines

INTERVENE when:
  1. User is about to start a step that needs specific guidance
     (e.g., temperature setting, timing, technique)
  2. A safety hazard is visible (sharp objects near edge, hot surfaces,
     electrical cords near water, etc.)
  3. The transition suggests the user might be confused or about to make
     a mistake (skipping steps, wrong order)
  4. The next step has a common error that proactive guidance can prevent

DO NOT intervene when:
  1. The transition is trivial (e.g., picking up a utensil)
  2. The user is clearly in flow and the next step is obvious
  3. The action is simple and self-explanatory
  4. Intervening would be distracting rather than helpful

## Output

Respond with ONLY valid JSON:
{
  "should_intervene": true/false,
  "intervention_type": "step_instruction" | "safety_warning" | "contextual_tip" | "error_correction" | "none",
  "intervention_mode": "anticipatory" | "reactive" | "none",
  "content": "The actual text to show on AR HUD (1-2 sentences, max 30 words)" | "",
  "priority": "low" | "medium" | "high",
  "rationale": "Brief explanation of why you made this decision",
  "frame_description": "What you see in the frame (1 sentence)",
  "visible_hazards": ["list", "of", "hazards"] or [],
  "visible_objects": ["key", "objects", "visible"]
}

Be realistic and strict. A good proactive system intervenes only when it
adds genuine value. Over-intervention is worse than under-intervention."""


def _build_gt_prompt(
    prev_action: str,
    next_action: str,
    recipe: str,
    action_sequence_context: str,
    boundary_index: int,
    total_boundaries: int,
) -> str:
    """Build the annotation prompt for a single boundary."""
    return f"""## Annotation Task

**Recipe**: {recipe}
**Progress**: Boundary {boundary_index + 1} of {total_boundaries}

**Action just completed**: {prev_action}
**Action about to begin**: {next_action}

**Surrounding action sequence** (for context):
{action_sequence_context}

Look at the attached frame (taken at the exact transition point between
the two actions above) and decide: should the AR system intervene here?

Respond with JSON only."""


# ── Generator ────────────────────────────────────────────────────────

class GTGenerator:
    """Generates ground truth intervention annotations using a strong VLM.

    Supports checkpoint/resume: saves progress after each boundary so
    interrupted runs can be resumed without re-annotating completed boundaries.
    """

    def __init__(
        self,
        model_spec: str = "gemini:gemini-3-flash",
        temperature: float = 0.2,
        egtea_data_root: str = "data/EGTEA_Gaze_Plus",
        delay_between_calls: float = 2.0,
        fallback_model_spec: Optional[str] = None,
    ):
        from agent.src.llm.factory import create_client
        from agent.ar_proactive.data.egtea_loader import EGTEALoader

        # Vertex AI: use more retries + longer backoff to handle rate limit (429) bursts
        is_vertex = model_spec.startswith("vertex:")
        self.client = create_client(
            model_spec,
            max_retries=6 if is_vertex else 3,
            retry_delay=3.0 if is_vertex else 1.0,
        )
        self._primary_model_spec = model_spec
        self._fallback_model_spec = fallback_model_spec
        self._fallback_client = None  # lazy-init on first quota error
        self._using_fallback = False
        self.loader = EGTEALoader(egtea_data_root)
        self.temperature = temperature
        self.delay = delay_between_calls

        info = model_spec
        if fallback_model_spec:
            info += f" (fallback: {fallback_model_spec})"
        logger.info(f"GTGenerator initialized with {info} (delay={delay_between_calls}s)")

    def generate_session_gt(
        self,
        session_name: str,
        max_boundaries: Optional[int] = None,
        context_window: int = 3,
        checkpoint_path: Optional[Path] = None,
    ) -> dict:
        """Generate ground truth for all action boundaries in a session.

        Args:
            session_name: EGTEA session name (e.g., "P01-R01-PastaSalad").
            max_boundaries: Limit number of boundaries to annotate (None = all).
            context_window: Number of surrounding actions to show as context.
            checkpoint_path: Path to save/resume checkpoint. If exists, resumes.

        Returns:
            Dict with session info, annotations, and stats.
        """
        session = self.loader.get_session(session_name)
        actions = session.actions

        if len(actions) < 2:
            raise ValueError(f"Session {session_name} has < 2 actions, no boundaries")

        # Identify action boundaries (where action label changes)
        boundaries = []
        for i in range(1, len(actions)):
            if actions[i].action_label != actions[i - 1].action_label:
                boundaries.append(i)

        if max_boundaries:
            boundaries = boundaries[:max_boundaries]

        # Resume from checkpoint if available
        annotations = []
        start_boundary = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    ckpt = json.load(f)
                existing = ckpt.get("annotations", [])
                annotations = [self._dict_to_annotation(a) for a in existing]
                start_boundary = len(existing)
                logger.info(
                    f"Resuming {session_name} from boundary {start_boundary+1} "
                    f"({len(existing)} already done)"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")

        logger.info(
            f"Session {session_name}: {len(actions)} actions, "
            f"{len(boundaries)} boundaries to annotate "
            f"(starting from {start_boundary+1})"
        )

        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5

        for bi in range(start_boundary, len(boundaries)):
            boundary_idx = boundaries[bi]
            prev_action = actions[boundary_idx - 1]
            next_action = actions[boundary_idx]

            # Build action context window
            ctx_start = max(0, boundary_idx - context_window)
            ctx_end = min(len(actions), boundary_idx + context_window + 1)
            context_lines = []
            for j in range(ctx_start, ctx_end):
                marker = " >>>" if j == boundary_idx else "    "
                context_lines.append(
                    f"{marker} [{actions[j].start_sec:.1f}s-{actions[j].end_sec:.1f}s] "
                    f"{actions[j].action_label}"
                )
            action_context = "\n".join(context_lines)

            # Extract frame at boundary
            boundary_time_ms = next_action.start_time_ms
            frame_b64 = self._extract_frame_at(session, boundary_time_ms)

            if frame_b64 is None:
                logger.warning(f"  Boundary {bi+1}: no frame at {boundary_time_ms}ms, skipping")
                continue

            # Build prompt
            prompt = _build_gt_prompt(
                prev_action=prev_action.action_label,
                next_action=next_action.action_label,
                recipe=session.recipe,
                action_sequence_context=action_context,
                boundary_index=bi,
                total_boundaries=len(boundaries),
            )

            # Rate limiting
            if bi > start_boundary and self.delay > 0:
                time.sleep(self.delay)

            # Call VLM
            try:
                annotation = self._annotate_boundary(
                    prompt=prompt,
                    frame_b64=frame_b64,
                    boundary_index=bi,
                    timestamp_sec=boundary_time_ms / 1000.0,
                    prev_action=prev_action.action_label,
                    next_action=next_action.action_label,
                    recipe=session.recipe,
                    session=session_name,
                )
                annotations.append(annotation)
                consecutive_failures = 0

                decision = "INTERVENE" if annotation.should_intervene else "SKIP"
                logger.info(
                    f"  Boundary {bi+1}/{len(boundaries)}: "
                    f"[{prev_action.action_label}] → [{next_action.action_label}] "
                    f"= {decision}"
                    + (f" ({annotation.intervention_type})" if annotation.should_intervene else "")
                )

                # Save checkpoint after each successful annotation
                if checkpoint_path:
                    self._save_checkpoint(
                        checkpoint_path, session_name, session, boundaries,
                        annotations, start_time,
                    )

            except Exception as e:
                consecutive_failures += 1
                err_str = str(e)
                logger.error(f"  Boundary {bi+1}: annotation failed: {e}")

                # Detect proxy/connection failure vs. content parsing failure
                is_connection_error = any(kw in err_str for kw in [
                    "Connection refused", "Connection aborted",
                    "RemoteDisconnected", "Max retries exceeded",
                    "NewConnectionError", "ConnectionError",
                ])

                if consecutive_failures >= max_consecutive_failures:
                    if is_connection_error:
                        # Proxy is down — wait for recovery instead of skipping session
                        logger.warning(
                            f"  Proxy appears down. Waiting for recovery "
                            f"(will retry every 60s)..."
                        )
                        recovered = self._wait_for_proxy_recovery(
                            max_wait_sec=1800,  # wait up to 30 min
                        )
                        if recovered:
                            logger.info("  Proxy recovered, resuming...")
                            consecutive_failures = 0
                            continue
                        else:
                            logger.error("  Proxy did not recover in 30min, stopping batch.")
                            break
                    else:
                        logger.error(
                            f"  {max_consecutive_failures} consecutive failures, "
                            f"stopping session. Use --resume to continue later."
                        )
                        break

                # Back off longer on repeated failures
                backoff = min(30, self.delay * (2 ** consecutive_failures))
                logger.info(f"  Backing off {backoff:.0f}s before next attempt...")
                time.sleep(backoff)

        elapsed = time.time() - start_time
        result = self._build_result(session_name, session, boundaries, annotations, elapsed)

        # Final save
        if checkpoint_path:
            self._save_checkpoint(
                checkpoint_path, session_name, session, boundaries,
                annotations, start_time,
            )

        return result

    def _wait_for_proxy_recovery(self, max_wait_sec: int = 1800, poll_sec: int = 60) -> bool:
        """Block until proxy is reachable again or timeout.

        Returns True if proxy recovered, False if timed out.
        """
        import urllib.request
        proxy_url = None
        if hasattr(self.client, 'config') and self.client.config.proxy:
            proxy_url = self.client.config.proxy
        else:
            # Try to detect from env
            import os
            proxy_url = os.environ.get("GEMINI_PROXY")

        if not proxy_url:
            logger.warning("Cannot determine proxy URL for health check.")
            return False

        # Extract host:port
        host_port = proxy_url.replace("http://", "").replace("https://", "").split("/")[0]
        host, port_str = host_port.rsplit(":", 1)
        port = int(port_str)

        waited = 0
        while waited < max_wait_sec:
            try:
                import socket
                with socket.create_connection((host, port), timeout=5):
                    logger.info(f"  Proxy {proxy_url} is reachable again!")
                    return True
            except OSError:
                logger.info(f"  Proxy still down, waiting {poll_sec}s... ({waited}s elapsed)")
                time.sleep(poll_sec)
                waited += poll_sec

        return False

    def generate_batch_gt(
        self,
        session_names: Optional[list[str]] = None,
        max_boundaries_per_session: Optional[int] = None,
        context_window: int = 3,
        output_dir: Path = Path("output/ar_proactive/gt"),
        checkpoint_subdir: str = "checkpoints",
    ) -> dict:
        """Generate GT for multiple sessions with checkpoint support.

        Args:
            session_names: List of sessions to process (None = all).
            max_boundaries_per_session: Limit per session.
            context_window: Surrounding action context.
            output_dir: Directory for checkpoints and results.

        Returns:
            Aggregate stats dict.
        """
        if session_names is None:
            session_names = self.loader.list_sessions()

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / checkpoint_subdir
        checkpoint_dir.mkdir(exist_ok=True)

        all_results = []
        total_start = time.time()

        # Check which sessions are already complete
        done_sessions = set()
        for name in session_names:
            final_path = output_dir / f"gt_{name}.json"
            if final_path.exists():
                try:
                    with open(final_path) as f:
                        data = json.load(f)
                    if data.get("annotated_boundaries") == data.get("total_boundaries"):
                        done_sessions.add(name)
                except Exception:
                    pass

        remaining = [s for s in session_names if s not in done_sessions]
        logger.info(
            f"Batch GT: {len(session_names)} sessions total, "
            f"{len(done_sessions)} already complete, "
            f"{len(remaining)} remaining"
        )

        for si, session_name in enumerate(remaining):
            logger.info(
                f"\n{'='*50}\n"
                f"  Session {si+1}/{len(remaining)}: {session_name}\n"
                f"{'='*50}"
            )

            # Verify proxy is alive before starting a new session
            # Skip check when using Vertex AI (no proxy needed)
            import os
            has_proxy = (
                (hasattr(self.client, 'config') and getattr(self.client.config, 'proxy', None))
                or os.environ.get("GEMINI_PROXY")
            )
            if si > 0 and has_proxy:
                recovered = self._wait_for_proxy_recovery(max_wait_sec=1800, poll_sec=30)
                if not recovered:
                    logger.error("Proxy not available after 30min, stopping batch.")
                    break

            ckpt_path = checkpoint_dir / f"ckpt_{session_name}.json"
            final_path = output_dir / f"gt_{session_name}.json"

            try:
                result = self.generate_session_gt(
                    session_name=session_name,
                    max_boundaries=max_boundaries_per_session,
                    context_window=context_window,
                    checkpoint_path=ckpt_path,
                )

                # Save final result
                with open(final_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Clean up checkpoint if fully complete
                if result["annotated_boundaries"] == result["total_boundaries"]:
                    if ckpt_path.exists():
                        ckpt_path.unlink()

                all_results.append(result)

                logger.info(
                    f"  Done: {result['annotated_boundaries']}/{result['total_boundaries']} "
                    f"boundaries, {result['interventions']} interventions "
                    f"({result['intervention_rate']:.0%})"
                )

            except Exception as e:
                logger.error(f"  Session {session_name} failed: {e}")

        total_elapsed = time.time() - total_start

        # Aggregate stats
        agg = self._aggregate_results(all_results, total_elapsed)
        agg_path = output_dir / "batch_summary.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)

        return agg

    @staticmethod
    def _build_result(session_name, session, boundaries, annotations, elapsed):
        """Build result dict from annotations."""
        n_intervene = sum(1 for a in annotations if a.should_intervene)
        type_dist = {}
        for a in annotations:
            if a.should_intervene:
                type_dist[a.intervention_type] = type_dist.get(a.intervention_type, 0) + 1

        return {
            "session": session_name,
            "recipe": session.recipe,
            "participant": session.participant,
            "total_actions": len(session.actions),
            "total_boundaries": len(boundaries),
            "annotated_boundaries": len(annotations),
            "interventions": n_intervene,
            "intervention_rate": round(n_intervene / max(1, len(annotations)), 3),
            "type_distribution": type_dist,
            "generation_time_sec": round(elapsed, 2),
            "model": "gemini-3-flash",
            "annotations": [a.to_dict() for a in annotations],
        }

    def _save_checkpoint(self, path, session_name, session, boundaries, annotations, start_time):
        """Save current progress as checkpoint."""
        elapsed = time.time() - start_time
        result = self._build_result(session_name, session, boundaries, annotations, elapsed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _dict_to_annotation(d: dict) -> GTAnnotation:
        """Reconstruct GTAnnotation from dict (for checkpoint resume)."""
        return GTAnnotation(
            boundary_index=d["boundary_index"],
            timestamp_sec=d["timestamp_sec"],
            prev_action=d["prev_action"],
            next_action=d["next_action"],
            recipe=d["recipe"],
            session=d["session"],
            should_intervene=d.get("should_intervene", False),
            intervention_type=d.get("intervention_type", ""),
            intervention_mode=d.get("intervention_mode", ""),
            content=d.get("content", ""),
            priority=d.get("priority", "medium"),
            rationale=d.get("rationale", ""),
            frame_description=d.get("frame_description", ""),
            visible_hazards=d.get("visible_hazards", []),
            visible_objects=d.get("visible_objects", []),
        )

    @staticmethod
    def _aggregate_results(results: list[dict], total_elapsed: float) -> dict:
        """Aggregate stats across multiple sessions."""
        total_boundaries = sum(r["total_boundaries"] for r in results)
        total_annotated = sum(r["annotated_boundaries"] for r in results)
        total_interventions = sum(r["interventions"] for r in results)

        type_dist = {}
        for r in results:
            for t, c in r.get("type_distribution", {}).items():
                type_dist[t] = type_dist.get(t, 0) + c

        return {
            "total_sessions": len(results),
            "total_boundaries": total_boundaries,
            "total_annotated": total_annotated,
            "total_interventions": total_interventions,
            "overall_intervention_rate": round(
                total_interventions / max(1, total_annotated), 3
            ),
            "type_distribution": type_dist,
            "total_time_sec": round(total_elapsed, 2),
            "avg_time_per_session_sec": round(
                total_elapsed / max(1, len(results)), 2
            ),
            "per_session": [
                {
                    "session": r["session"],
                    "boundaries": r["total_boundaries"],
                    "annotated": r["annotated_boundaries"],
                    "interventions": r["interventions"],
                    "rate": r["intervention_rate"],
                    "types": r["type_distribution"],
                }
                for r in results
            ],
        }

    def _switch_to_fallback(self):
        """Switch from primary to fallback model (e.g., on quota exhaustion)."""
        if self._using_fallback or not self._fallback_model_spec:
            return False
        if self._fallback_client is None:
            import os
            from agent.src.llm.factory import create_client
            # If a fallback proxy is configured, inject it before creating the client
            fallback_proxy = os.environ.get("GEMINI_PROXY_FALLBACK")
            if fallback_proxy:
                os.environ["GEMINI_PROXY"] = fallback_proxy
                logger.info(f"Activating fallback proxy: {fallback_proxy}")
            self._fallback_client = create_client(self._fallback_model_spec)
        self.client = self._fallback_client
        self._using_fallback = True
        logger.warning(
            f"Switched to fallback model: {self._fallback_model_spec} "
            f"(primary quota exhausted)"
        )
        return True

    def _annotate_boundary(
        self,
        prompt: str,
        frame_b64: str,
        boundary_index: int,
        timestamp_sec: float,
        prev_action: str,
        next_action: str,
        recipe: str,
        session: str,
    ) -> GTAnnotation:
        """Annotate a single boundary using VLM, with auto-fallback on quota errors."""
        try:
            response = self.client.complete_with_images(
                prompt=prompt,
                images=[frame_b64],
                system_prompt=GT_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=4096,
            )
        except Exception as e:
            err_str = str(e)
            is_quota = any(kw in err_str for kw in [
                "quota", "QUOTA", "429", "exceeded your current quota",
                "resource_exhausted", "RESOURCE_EXHAUSTED",
            ])
            if is_quota and self._switch_to_fallback():
                # Retry with fallback model
                response = self.client.complete_with_images(
                    prompt=prompt,
                    images=[frame_b64],
                    system_prompt=GT_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
            else:
                raise

        data = self._parse_response(response)

        return GTAnnotation(
            boundary_index=boundary_index,
            timestamp_sec=timestamp_sec,
            prev_action=prev_action,
            next_action=next_action,
            recipe=recipe,
            session=session,
            should_intervene=bool(data.get("should_intervene", False)),
            intervention_type=data.get("intervention_type", "none"),
            intervention_mode=data.get("intervention_mode", "none"),
            content=data.get("content", ""),
            priority=data.get("priority", "medium"),
            rationale=data.get("rationale", ""),
            frame_description=data.get("frame_description", ""),
            visible_hazards=data.get("visible_hazards", []),
            visible_objects=data.get("visible_objects", []),
        )

    def _extract_frame_at(
        self,
        session,
        time_ms: int,
    ) -> Optional[str]:
        """Extract a single frame at the given timestamp from session clips."""
        # Find the clip covering this timestamp
        for clip in session.clips:
            if clip.start_time_ms <= time_ms <= clip.end_time_ms:
                cap = cv2.VideoCapture(str(clip.path))
                if not cap.isOpened():
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                relative_ms = time_ms - clip.start_time_ms
                target_frame = int((relative_ms / 1000.0) * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return base64.b64encode(buf).decode()

        return None

    @staticmethod
    def _parse_response(response: str) -> dict:
        """Parse VLM response, handling thinking model output and truncation."""

        text = response.strip()
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find best JSON object (for thinking models)
        best_json = None
        best_fields = 0
        expected = {"should_intervene", "intervention_type", "content", "rationale"}

        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                parsed = json.loads(text[i:j+1])
                                n = len(expected & set(parsed.keys()))
                                if n > best_fields:
                                    best_json = parsed
                                    best_fields = n
                            except json.JSONDecodeError:
                                pass
                            break

        if best_json is not None:
            return best_json

        # Attempt to repair truncated JSON (from MAX_TOKENS cutoff)
        repaired = GTGenerator._repair_truncated_json(text)
        if repaired is not None:
            return repaired

        raise ValueError(f"Could not parse GT response: {text[:200]}")

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[dict]:
        """Try to repair JSON that was truncated by MAX_TOKENS.

        Strategy: find the last complete key-value pair, truncate there,
        and close the JSON object.
        """
        # Find the start of a JSON object
        start = text.find('{')
        if start < 0:
            return None

        fragment = text[start:]

        # Try progressively aggressive repairs
        for attempt in range(3):
            if attempt == 0:
                # Try closing with just "}"
                candidate = fragment.rstrip(',\n\r\t "') + '}'
            elif attempt == 1:
                # Truncate at last complete value (after a comma or closing bracket)
                # Find last comma that's followed by a key pattern
                last_good = -1
                for m in re.finditer(r',\s*"[^"]+"\s*:', fragment):
                    last_good = m.start()
                if last_good > 0:
                    candidate = fragment[:last_good] + '}'
                else:
                    continue
            else:
                # Find the last complete "key": value pair
                last_good = -1
                in_string = False
                for idx, c in enumerate(fragment):
                    if c == '"' and (idx == 0 or fragment[idx-1] != '\\'):
                        in_string = not in_string
                    if not in_string and c == ',':
                        last_good = idx
                if last_good > 0:
                    candidate = fragment[:last_good] + '}'
                else:
                    continue

            # Close any open arrays
            open_brackets = candidate.count('[') - candidate.count(']')
            if open_brackets > 0:
                # Truncate the incomplete array item and close
                last_bracket = candidate.rfind('[')
                if last_bracket > 0:
                    # Find if there's a complete array before truncation
                    candidate = candidate[:last_bracket] + '[]}'
                    open_brackets = candidate.count('[') - candidate.count(']')
                    for _ in range(open_brackets):
                        candidate += ']'
                    if candidate.count('{') > candidate.count('}'):
                        candidate += '}'

            try:
                parsed = json.loads(candidate)
                if "should_intervene" in parsed:
                    logger.warning(f"Repaired truncated JSON (attempt {attempt+1})")
                    return parsed
            except json.JSONDecodeError:
                continue

        return None


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ground truth intervention annotations",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="EGTEA session name (default: P01-R01-PastaSalad)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all sessions in batch mode",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Process specific list of sessions",
    )
    parser.add_argument(
        "--max-boundaries",
        type=int,
        default=None,
        help="Max boundaries to annotate per session (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini:gemini-3-flash",
        help="Model spec for annotation",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of surrounding actions for context",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between API calls (default: 2.0)",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Fallback model spec when primary quota exhausted (e.g. gemini:gemini-3-pro)",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        default="checkpoints",
        help="Checkpoint subdirectory name (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints (for batch mode)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (file for single, dir for batch)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    gen = GTGenerator(
        model_spec=args.model,
        delay_between_calls=args.delay,
        fallback_model_spec=args.fallback_model,
    )

    out_dir = Path(args.output) if args.output else Path("output/ar_proactive/gt")

    if args.batch or args.sessions:
        # Batch mode
        out_dir.mkdir(parents=True, exist_ok=True)
        session_list = args.sessions  # None = all sessions

        result = gen.generate_batch_gt(
            session_names=session_list,
            max_boundaries_per_session=args.max_boundaries,
            context_window=args.context_window,
            output_dir=out_dir,
            checkpoint_subdir=args.checkpoint_subdir,
        )

        print(f"\n{'=' * 60}")
        print(f"  BATCH GT GENERATION SUMMARY")
        print(f"  Sessions: {result['total_sessions']}")
        print(f"  Total boundaries: {result['total_annotated']}/{result['total_boundaries']}")
        print(f"  Interventions: {result['total_interventions']} "
              f"({result['overall_intervention_rate']:.0%})")
        print(f"  Type distribution: {result['type_distribution']}")
        print(f"  Time: {result['total_time_sec']:.1f}s "
              f"({result['avg_time_per_session_sec']:.1f}s/session)")
        print(f"{'=' * 60}")

        for s in result["per_session"]:
            status = "OK" if s["annotated"] == s["boundaries"] else "PARTIAL"
            print(f"  [{status}] {s['session']}: "
                  f"{s['annotated']}/{s['boundaries']} boundaries, "
                  f"{s['interventions']} interventions ({s['rate']:.0%})")

        print(f"\nSaved to: {out_dir}/")

    else:
        # Single session mode
        session_name = args.session or "P01-R01-PastaSalad"

        ckpt_dir = out_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"ckpt_{session_name}.json" if args.resume else None

        result = gen.generate_session_gt(
            session_name=session_name,
            max_boundaries=args.max_boundaries,
            context_window=args.context_window,
            checkpoint_path=ckpt_path,
        )

        # Save
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.output and not Path(args.output).is_dir():
            out_path = Path(args.output)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"gt_{session_name}_{ts}.json"

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  GT GENERATION: {session_name}")
        print(f"  Boundaries annotated: {result['annotated_boundaries']}/{result['total_boundaries']}")
        print(f"  Interventions: {result['interventions']} "
              f"({result['intervention_rate']:.0%})")
        print(f"  Type distribution: {result['type_distribution']}")
        print(f"  Time: {result['generation_time_sec']:.1f}s")
        print(f"{'=' * 60}")

        for a in result["annotations"]:
            decision = "INTERVENE" if a["should_intervene"] else "SKIP"
            print(f"\n  [{a['timestamp_sec']:.1f}s] {a['prev_action']} → {a['next_action']}")
            print(f"    Decision: {decision}")
            if a["should_intervene"]:
                print(f"    Type: {a['intervention_type']} ({a['intervention_mode']})")
                print(f"    Content: \"{a['content']}\"")
                print(f"    Priority: {a['priority']}")
            print(f"    Scene: {a['frame_description']}")
            if a["visible_hazards"]:
                print(f"    Hazards: {a['visible_hazards']}")
            print(f"    Rationale: {a['rationale']}")

        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
