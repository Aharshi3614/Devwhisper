"""Tests for pipeline stage order validation (issue #227).

Issue #227: Validate Pipeline Stage Order.

These tests confirm that:
  * The canonical pipeline stage order is defined and stable.
  * PipelineTracker correctly validates forward transitions.
  * PipelineTracker detects and rejects invalid transitions:
      - Unknown stage names
      - Out-of-order execution (going backwards)
      - Wrong first stage
  * The cache-hit fast path is still a valid sequence.
  * validate_stage_sequence() convenience helper behaves identically.
  * Integration: main.py imports the tracker correctly.
"""

import pytest

from pipeline_validator import (
    PIPELINE_STAGES,
    STAGE_LABELS,
    PipelineStageError,
    PipelineTracker,
    validate_stage_sequence,
)


# ---------------------------------------------------------------------------
# Canonical stage definition
# ---------------------------------------------------------------------------

def test_pipeline_stages_is_defined_and_ordered():
    """The canonical pipeline stage order must be a non-empty tuple."""
    assert isinstance(PIPELINE_STAGES, tuple)
    assert len(PIPELINE_STAGES) >= 5
    assert PIPELINE_STAGES[0] == "cache_lookup"
    assert PIPELINE_STAGES[-1] == "cache_insertion"


def test_pipeline_stages_are_unique():
    """No stage should appear twice in the canonical order."""
    assert len(PIPELINE_STAGES) == len(set(PIPELINE_STAGES))


def test_every_stage_has_a_label():
    """Every stage ID should have a human-readable label."""
    for stage in PIPELINE_STAGES:
        assert stage in STAGE_LABELS, f"Missing label for stage {stage!r}"
        assert isinstance(STAGE_LABELS[stage], str)
        assert STAGE_LABELS[stage].strip()


# ---------------------------------------------------------------------------
# PipelineTracker — valid transitions
# ---------------------------------------------------------------------------

def test_tracker_accepts_full_forward_sequence():
    """A complete forward walk through the pipeline should not raise."""
    tracker = PipelineTracker()
    for stage in PIPELINE_STAGES:
        tracker.enter(stage)
    assert tracker.is_complete()
    assert tracker.history == list(PIPELINE_STAGES)


def test_tracker_accepts_forward_skips():
    """Stages may be skipped (e.g., cache hit skips retrieval)."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("cache_insertion")  # skip retrieval, generation, post_processing
    assert tracker.last_stage == "cache_insertion"
    assert tracker.is_complete()


def test_tracker_first_stage_must_be_cache_lookup():
    """The first stage recorded must be cache_lookup."""
    tracker = PipelineTracker()
    with pytest.raises(PipelineStageError) as exc_info:
        tracker.enter("retrieval")
    assert exc_info.value.current_stage == "retrieval"
    assert exc_info.value.previous_stage is None
    assert "First stage must be" in exc_info.value.reason


def test_tracker_records_history_in_order():
    """The history list should reflect the order stages were entered."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    tracker.enter("generation")
    assert tracker.history == ["cache_lookup", "retrieval", "generation"]


def test_tracker_last_stage_property():
    """The last_stage property should track the most recent stage."""
    tracker = PipelineTracker()
    assert tracker.last_stage is None
    tracker.enter("cache_lookup")
    assert tracker.last_stage == "cache_lookup"
    tracker.enter("retrieval")
    assert tracker.last_stage == "retrieval"


def test_tracker_current_index():
    """The current_index property should return the canonical index."""
    tracker = PipelineTracker()
    assert tracker.current_index == -1
    tracker.enter("cache_lookup")
    assert tracker.current_index == 0
    tracker.enter("retrieval")
    assert tracker.current_index == 1
    tracker.enter("cache_insertion")
    assert tracker.current_index == 4


# ---------------------------------------------------------------------------
# PipelineTracker — invalid transitions
# ---------------------------------------------------------------------------

def test_tracker_rejects_unknown_stage():
    """An unknown stage name should raise PipelineStageError."""
    tracker = PipelineTracker()
    with pytest.raises(PipelineStageError) as exc_info:
        tracker.enter("nonexistent_stage")
    assert exc_info.value.current_stage == "nonexistent_stage"
    assert "Unknown stage" in exc_info.value.reason


def test_tracker_rejects_backward_transition():
    """Going backwards in the pipeline should raise."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    with pytest.raises(PipelineStageError) as exc_info:
        tracker.enter("cache_lookup")  # going back
    assert exc_info.value.current_stage == "cache_lookup"
    assert exc_info.value.previous_stage == "retrieval"
    assert "cannot run after" in exc_info.value.reason


def test_tracker_allows_generation_after_cache_lookup():
    """Generation after cache_lookup is a valid forward skip (no retrieval)."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("generation")  # forward skip — allowed
    assert tracker.last_stage == "generation"


def test_tracker_rejects_post_processing_before_generation():
    """Post-processing cannot run before generation."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    # Skip generation, go straight to post_processing — this is a valid
    # forward skip and should NOT raise.
    tracker.enter("post_processing")
    assert tracker.last_stage == "post_processing"


def test_tracker_rejects_cache_insertion_before_cache_lookup():
    """The final stage cannot run without cache_lookup first."""
    tracker = PipelineTracker()
    with pytest.raises(PipelineStageError) as exc_info:
        tracker.enter("cache_insertion")
    assert "First stage must be" in exc_info.value.reason


def test_tracker_rejects_re_entry_after_later_stage():
    """Once a later stage has run, you cannot re-enter an earlier one."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    tracker.enter("generation")
    with pytest.raises(PipelineStageError):
        tracker.enter("retrieval")  # going back


# ---------------------------------------------------------------------------
# PipelineStageError attributes
# ---------------------------------------------------------------------------

def test_pipeline_stage_error_attributes():
    """The error should expose current_stage, previous_stage, and reason."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    try:
        tracker.enter("cache_lookup")
    except PipelineStageError as e:
        assert e.current_stage == "cache_lookup"
        assert e.previous_stage == "retrieval"
        assert isinstance(e.reason, str) and e.reason
        assert "cannot run after" in str(e)


# ---------------------------------------------------------------------------
# reset() and is_complete()
# ---------------------------------------------------------------------------

def test_tracker_reset_clears_state():
    """reset() should clear the history and last stage."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    tracker.reset()
    assert tracker.history == []
    assert tracker.last_stage is None
    assert tracker.current_index == -1
    assert not tracker.is_complete()


def test_tracker_is_complete_only_after_final_stage():
    """is_complete() should return True only after the final stage."""
    tracker = PipelineTracker()
    assert not tracker.is_complete()
    tracker.enter("cache_lookup")
    assert not tracker.is_complete()
    tracker.enter("retrieval")
    assert not tracker.is_complete()
    tracker.enter("generation")
    assert not tracker.is_complete()
    tracker.enter("post_processing")
    assert not tracker.is_complete()
    tracker.enter("cache_insertion")
    assert tracker.is_complete()


def test_tracker_summary():
    """summary() should return a diagnostic dict."""
    tracker = PipelineTracker()
    tracker.enter("cache_lookup")
    tracker.enter("retrieval")
    summary = tracker.summary()
    assert summary["stages_recorded"] == ["cache_lookup", "retrieval"]
    assert summary["last_stage"] == "retrieval"
    assert summary["is_complete"] is False
    assert summary["expected_order"] == list(PIPELINE_STAGES)


# ---------------------------------------------------------------------------
# validate_stage_sequence() convenience function
# ---------------------------------------------------------------------------

def test_validate_stage_sequence_accepts_full_pipeline():
    """A full forward walk should not raise."""
    validate_stage_sequence(list(PIPELINE_STAGES))


def test_validate_stage_sequence_accepts_cache_hit_path():
    """The cache-hit fast path (1 → 5) should be valid."""
    validate_stage_sequence(["cache_lookup", "cache_insertion"])


def test_validate_stage_sequence_rejects_out_of_order():
    """An out-of-order sequence should raise."""
    with pytest.raises(PipelineStageError):
        validate_stage_sequence(["retrieval", "cache_lookup"])


def test_validate_stage_sequence_rejects_unknown_stage():
    """An unknown stage should raise."""
    with pytest.raises(PipelineStageError):
        validate_stage_sequence(["cache_lookup", "bogus_stage"])


def test_validate_stage_sequence_rejects_empty():
    """An empty sequence should not raise (nothing to validate)."""
    validate_stage_sequence([])


def test_validate_stage_sequence_rejects_wrong_first_stage():
    """Starting with a non-first stage should raise."""
    with pytest.raises(PipelineStageError) as exc_info:
        validate_stage_sequence(["generation"])
    assert "First stage must be" in exc_info.value.reason


# ---------------------------------------------------------------------------
# Integration: main.py pipeline uses the tracker correctly
# ---------------------------------------------------------------------------

def test_main_pipeline_imports_tracker():
    """main.py should import PipelineTracker from pipeline_validator."""
    import main
    assert hasattr(main, "PipelineTracker")
    assert hasattr(main, "PipelineStageError")


def test_pipeline_validator_module_loads_cleanly():
    """The pipeline_validator module should import without errors."""
    import pipeline_validator
    assert hasattr(pipeline_validator, "PIPELINE_STAGES")
    assert hasattr(pipeline_validator, "PipelineTracker")
    assert hasattr(pipeline_validator, "PipelineStageError")
    assert hasattr(pipeline_validator, "validate_stage_sequence")
