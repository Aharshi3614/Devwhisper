"""pipeline_validator.py — Validate request-processing pipeline stage order.

Issue #227: Validate Pipeline Stage Order.

The DevWhisper request-processing pipeline has a well-defined sequence of
stages (see handlers.process_query_pipeline):

    Stage 1: Cache Lookup
    Stage 2: Retrieval (Hybrid Search)
    Stage 3: Command Routing or LLM Generation
    Stage 4: Post-processing & Attribution
    Stage 5: Cache Insertion & Memory Update

Stages may be skipped in legitimate cases (e.g., a cache hit at Stage 1
means Stages 2–4 don't run, and Stage 5 still runs to update memory).
However, stages MUST NOT execute out of order — e.g., Stage 3 cannot
run before Stage 2, and Stage 5 cannot run before Stage 1.

This module provides:
  - PIPELINE_STAGES: the canonical ordered list of stage names
  - PipelineTracker: records stage transitions and validates them
  - PipelineStageError: raised when an invalid transition is detected
  - validate_stage_sequence(): convenience function for validating a
    pre-recorded list of stages

The validator is intentionally framework-agnostic — it can be used by
handlers.py, tests, or any monitoring/diagnostic code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical pipeline definition
# ---------------------------------------------------------------------------

# The expected execution order of pipeline stages. Each stage is identified
# by a stable string ID so it can be referenced from logs, tests, and the
# frontend timeline without being coupled to display labels.
#
# These IDs mirror the stages documented in handlers.process_query_pipeline()
# and the frontend ProcessingTimeline component (frontend/src/App.jsx):
#   request_received → retrieval → generation → completion
#
# The backend pipeline is slightly more granular (it splits cache lookup
# and post-processing into their own stages), so this module defines the
# authoritative backend ordering.
PIPELINE_STAGES: tuple[str, ...] = (
    "cache_lookup",        # Stage 1: Check the response cache
    "retrieval",           # Stage 2: Hybrid (vector + BM25) retrieval
    "generation",          # Stage 3: Command routing or LLM generation
    "post_processing",     # Stage 4: Attribution, formatting, sanitization
    "cache_insertion",     # Stage 5: Cache the answer + update session memory
)

# Human-readable labels for logging / error messages.
STAGE_LABELS: dict[str, str] = {
    "cache_lookup":    "Cache Lookup",
    "retrieval":       "Retrieval (Hybrid Search)",
    "generation":      "Command Routing or LLM Generation",
    "post_processing": "Post-processing & Attribution",
    "cache_insertion": "Cache Insertion & Memory Update",
}

# Stages that are allowed to be skipped (e.g., cache hit skips retrieval).
# Every stage is in this set by default — skipping is a legitimate
# optimization. What is NOT legitimate is running a stage BEFORE its
# predecessor has either run or been skipped.
_SKIPPABLE_STAGES = set(PIPELINE_STAGES)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class PipelineStageError(RuntimeError):
    """Raised when an invalid pipeline stage transition is detected.

    Attributes:
        current_stage: The stage that was about to run.
        previous_stage: The most recent stage that ran before it (or None).
        reason: Human-readable explanation of why the transition is invalid.
    """

    def __init__(
        self,
        current_stage: str,
        previous_stage: Optional[str],
        reason: str,
    ) -> None:
        self.current_stage = current_stage
        self.previous_stage = previous_stage
        self.reason = reason
        super().__init__(
            f"Invalid pipeline stage transition: {reason} "
            f"(current={current_stage!r}, previous={previous_stage!r})"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class PipelineTracker:
    """Records pipeline stage transitions and validates their order.

    Usage:
        tracker = PipelineTracker()
        tracker.enter("cache_lookup")     # Stage 1
        tracker.enter("retrieval")        # Stage 2 — OK, comes after cache_lookup
        tracker.enter("cache_lookup")     # raises PipelineStageError — can't go back

    The tracker is intentionally stateful and not thread-safe; each request
    should instantiate its own tracker (or call reset() between requests).
    """

    #: The canonical stage order, copied here so tests can mutate the
    #: tracker's view without touching the module-level constant.
    stages: tuple[str, ...] = PIPELINE_STAGES

    #: Stages that have been recorded so far, in execution order.
    history: list[str] = field(default_factory=list)

    #: The most recent stage that was recorded (or None if none yet).
    _last: Optional[str] = field(default=None, repr=False)

    @property
    def last_stage(self) -> Optional[str]:
        """Return the most recently recorded stage, or None."""
        return self._last

    @property
    def current_index(self) -> int:
        """Return the index of the last stage in the canonical order.

        Returns -1 if no stage has been recorded yet.
        """
        if self._last is None:
            return -1
        try:
            return self.stages.index(self._last)
        except ValueError:
            return -1

    def enter(self, stage: str) -> None:
        """Record that a stage is about to execute, validating its order.

        Validates:
          1. The stage name is a known pipeline stage.
          2. The stage comes AFTER the previously-recorded stage in the
             canonical order (or is the same stage being re-entered, which
             is allowed only if no later stage has run yet).
          3. The stage is not being re-entered after a later stage has
             already run (e.g., cannot go back to cache_lookup after
             generation has started).

        Args:
            stage: The stage ID about to execute.

        Raises:
            PipelineStageError: If the transition is invalid.
        """
        if stage not in self.stages:
            raise PipelineStageError(
                current_stage=stage,
                previous_stage=self._last,
                reason=f"Unknown stage {stage!r} — not in canonical pipeline order",
            )

        if self._last is None:
            # First stage — must be the first stage in the canonical order.
            if stage != self.stages[0]:
                raise PipelineStageError(
                    current_stage=stage,
                    previous_stage=None,
                    reason=(
                        f"First stage must be {self.stages[0]!r}, got {stage!r}"
                    ),
                )
        else:
            last_idx = self.stages.index(self._last)
            new_idx = self.stages.index(stage)

            if new_idx < last_idx:
                # Going backwards — not allowed.
                raise PipelineStageError(
                    current_stage=stage,
                    previous_stage=self._last,
                    reason=(
                        f"Stage {stage!r} ({STAGE_LABELS.get(stage, stage)}) "
                        f"cannot run after {self._last!r} "
                        f"({STAGE_LABELS.get(self._last, self._last)}) — "
                        f"pipeline must move forward"
                    ),
                )

        self.history.append(stage)
        self._last = stage

    def reset(self) -> None:
        """Clear all recorded stages. Useful for reusing the tracker."""
        self.history.clear()
        self._last = None

    def is_complete(self) -> bool:
        """Return True if the final stage has been recorded."""
        return self._last == self.stages[-1]

    def summary(self) -> dict:
        """Return a diagnostic summary of the recorded pipeline run."""
        return {
            "stages_recorded": list(self.history),
            "last_stage": self._last,
            "is_complete": self.is_complete(),
            "expected_order": list(self.stages),
        }


# ---------------------------------------------------------------------------
# Indexing Pipeline Definition
# ---------------------------------------------------------------------------

INDEXING_PIPELINE_STAGES: tuple[str, ...] = (
    "collect_files",
    "extract_symbols",
    "generate_chunks",
    "embed_and_upsert",
    "persist_cache",
)


@dataclass
class IndexingPipelineTracker(PipelineTracker):
    """Pipeline tracker specialized for codebase indexing stages."""

    stages: tuple[str, ...] = INDEXING_PIPELINE_STAGES



# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_stage_sequence(stages: list[str]) -> None:
    """Validate that a list of stage IDs is in correct pipeline order.

    Args:
        stages: List of stage IDs in the order they were executed.

    Raises:
        PipelineStageError: If any stage is unknown or out of order.
    """
    tracker = PipelineTracker()
    for stage in stages:
        tracker.enter(stage)


def validate_indexing_sequence(stages: list[str]) -> None:
    """Validate that an indexing stage sequence executes forward."""
    tracker = IndexingPipelineTracker()
    for stage in stages:
        tracker.enter(stage)


__all__ = [
    "PIPELINE_STAGES",
    "STAGE_LABELS",
    "INDEXING_PIPELINE_STAGES",
    "PipelineStageError",
    "PipelineTracker",
    "IndexingPipelineTracker",
    "validate_stage_sequence",
    "validate_indexing_sequence",
]

