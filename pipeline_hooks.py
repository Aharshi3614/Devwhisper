"""pipeline_hooks.py — Standardized Pipeline Hooks for DevWhisper processing stages.

Provides extensible lifecycle hooks allowing custom logic pre- and post-processing across indexing, retrieval, and LLM generation stages.
"""

from typing import Callable, Dict, List, Any
from logger import logger

PreHookFn = Callable[[str, Dict[str, Any]], None]
PostHookFn = Callable[[str, Dict[str, Any], Any], None]

class PipelineHookRegistry:
    def __init__(self):
        self._pre_hooks: Dict[str, List[PreHookFn]] = {}
        self._post_hooks: Dict[str, List[PostHookFn]] = {}

    def register_pre_hook(self, stage: str, fn: PreHookFn) -> None:
        if stage not in self._pre_hooks:
            self._pre_hooks[stage] = []
        self._pre_hooks[stage].append(fn)

    def register_post_hook(self, stage: str, fn: PostHookFn) -> None:
        if stage not in self._post_hooks:
            self._post_hooks[stage] = []
        self._post_hooks[stage].append(fn)

    def execute_pre_hooks(self, stage: str, payload: Dict[str, Any]) -> None:
        hooks = self._pre_hooks.get(stage, [])
        for fn in hooks:
            try:
                fn(stage, payload)
            except Exception as e:
                logger.error(f"Error executing pre-hook for stage {stage}: {e}")

    def execute_post_hooks(self, stage: str, payload: Dict[str, Any], result: Any) -> None:
        hooks = self._post_hooks.get(stage, [])
        for fn in hooks:
            try:
                fn(stage, payload, result)
            except Exception as e:
                logger.error(f"Error executing post-hook for stage {stage}: {e}")

# Global singleton hook registry
hook_registry = PipelineHookRegistry()
