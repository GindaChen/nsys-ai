"""Shared cache-build progress for Textual apps.

``build_cache`` accepts a progress callback so Textual callers can silence the
``\\r`` stderr redraw (Textual's stderr pretends to be a tty). The three apps
share one Message type and one helper that posts it from a worker thread —
replacing the three identical no-ops that only silenced stderr while the build
still ran on the message loop (#332).
"""

from __future__ import annotations

from textual.app import App
from textual.message import Message
from textual.worker import NoActiveWorker, get_current_worker


class CacheBuildProgress(Message):
    """One step of an in-progress parquet cache build."""

    def __init__(self, label: str, step: int, total: int) -> None:
        super().__init__()
        self.label = label
        self.step = step
        self.total = total


def post_cache_progress(app: App, label: str, step: int, total: int) -> None:
    """Progress callback for ``build_cache``: post ``CacheBuildProgress``.

    Honours worker cancellation — a cancelled thread worker cannot be
    interrupted, so the callback itself must stop posting (#311 / #332).
    Also skips when the app has already left ``is_running``, so a progress
    step that races quit does not enqueue UI work against a torn-down screen.
    """
    try:
        worker = get_current_worker()
    except NoActiveWorker:
        worker = None
    if worker is not None and worker.is_cancelled:
        return
    if not app.is_running:
        return
    app.post_message(CacheBuildProgress(label, step, total))


def format_cache_progress(label: str, step: int, total: int) -> str:
    """Subtitle text for a cache-build progress step."""
    return f"Building cache [{step}/{total}] {label}"
