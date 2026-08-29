"""indexing_job_manager.py — Thread-safe indexing job manager and cancellation coordinator.

This module coordinates asynchronous codebase indexing tasks. It ensures that concurrent
indexing requests for the same repository do not collide, tracks job lifecycles with granular
progress state, and enables cooperative job cancellation.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class JobStatus(str, enum.Enum):
    """Lifecycle statuses for codebase indexing jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IndexingJob:
    """Represents an indexing job and its execution state."""

    id: str
    repo_id: str
    is_dry_run: bool = False
    status: JobStatus = JobStatus.PENDING
    percent: int = 0
    message: str = "Job queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    indexed_files: int = 0
    total_files: int = 0
    _cancellation_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> bool:
        """Signal the job to cancel if still active."""
        if self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        self._cancellation_event.set()
        self.status = JobStatus.CANCELLED
        self.message = "Job cancelled by user"
        self.finished_at = time.time()
        return True

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        return self._cancellation_event.is_set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert job state to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "is_dry_run": self.is_dry_run,
            "status": self.status.value,
            "percent": self.percent,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "indexed_files": self.indexed_files,
            "total_files": self.total_files,
        }


class IndexingJobManager:
    """Central repository indexing coordinator."""

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self._jobs: Dict[str, IndexingJob] = {}
        self._lock = threading.Lock()

    def create_job(self, repo_id: str, is_dry_run: bool = False) -> IndexingJob:
        """Create and register a new indexing job."""
        job_id = str(uuid.uuid4())
        job = IndexingJob(
            id=job_id,
            repo_id=repo_id,
            is_dry_run=is_dry_run,
        )
        with self._lock:
            self._jobs[job_id] = job
            # Keep history within bounds
            if len(self._jobs) > self.max_history:
                oldest_key = next(iter(self._jobs))
                del self._jobs[oldest_key]
        return job

    def get_job(self, job_id: str) -> Optional[IndexingJob]:
        """Retrieve job by its unique identifier."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            return job.cancel()

    def get_active_job_for_repo(self, repo_id: str) -> Optional[IndexingJob]:
        """Check if an active (pending or running) job is already processing the repo."""
        with self._lock:
            for job in self._jobs.values():
                if job.repo_id == repo_id and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    return job
            return None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recently submitted jobs."""
        with self._lock:
            all_jobs = list(self._jobs.values())
            all_jobs.sort(key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in all_jobs[:limit]]

    def clear(self) -> None:
        """Clear all registered jobs."""
        with self._lock:
            self._jobs.clear()


# Global singleton instance
job_manager = IndexingJobManager()
