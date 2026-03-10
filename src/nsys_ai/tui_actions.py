"""
tui_actions.py - Map parsed tool-call actions to TUI Python APIs (,).

When the agent returns navigate_to_kernel or zoom_to_time_range, the TUI (curses/Textual)
calls execute_tui_action(action_dict, app) so the same tool schema drives both Web (JS)
and TUI (Python). The app object should implement the methods below; missing methods
are no-ops.
"""

from typing import Any


def execute_tui_action(action_dict: dict, app: Any) -> bool:
    """
    Execute a single parsed action (navigate_to_kernel or zoom_to_time_range) on the TUI app.

    The app is expected to provide:
      - scroll_to_kernel(target_name: str, occurrence_index: int = 1) for navigate_to_kernel
      - zoom_to_time_range(start_s: float, end_s: float) for zoom_to_time_range

    Returns True if the action was handled, False if unknown or app has no handler.
    """
    if not action_dict or not isinstance(action_dict, dict):
        return False
    action_type = action_dict.get("type")
    if action_type == "navigate_to_kernel":
        target = action_dict.get("target_name")
        if not target:
            return False
        occurrence = action_dict.get("occurrence_index", 1)
        handler = getattr(app, "scroll_to_kernel", None)
        if callable(handler):
            handler(target, occurrence)
            return True
        return False
    if action_type == "zoom_to_time_range":
        start_s = action_dict.get("start_s")
        end_s = action_dict.get("end_s")
        if start_s is None or end_s is None:
            return False
        handler = getattr(app, "zoom_to_time_range", None)
        if callable(handler):
            handler(float(start_s), float(end_s))
            return True
        return False
    return False
