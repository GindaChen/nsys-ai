"""
main_page.py — The entry point for nsys-ai.

`nsys-ai`      → Interactive TUI: find profiles, pick action, launch.
`nsys-ai help` → Print getting-started guide to stdout.
"""
import curses
import glob
import os
import sys
from typing import Optional


BANNER = r"""
  ┌─────────────────────────────────────────────┐
  │              🔬  nsys-ai                     │
  │   AI-powered GPU profile analysis            │
  │                                              │
  │   Navigate timelines · Diagnose bottlenecks  │
  │   Explore NVTX trees · Run analysis skills   │
  └─────────────────────────────────────────────┘
"""

ACTIONS = [
    ("timeline", "🕐", "Timeline TUI  — Perfetto-style horizontal view"),
    ("tui",      "🌲", "Tree TUI      — NVTX hierarchy browser"),
    ("summary",  "📊", "Summary       — Kernel stats & auto-commentary"),
    ("info",     "ℹ️ ", "Info          — Profile metadata & GPU hardware"),
    ("skill",    "🧩", "Skills        — Run analysis skills"),
    ("agent",    "🤖", "Agent         — AI auto-analysis"),
    ("web",      "🌐", "Web UI        — Browser-based viewer"),
]


def _find_profiles() -> list[str]:
    """Find .sqlite profiles in current directory tree (max 1 level deep)."""
    profiles = set()
    for pattern in ["*.sqlite", "*/*.sqlite"]:
        profiles.update(glob.glob(pattern))
    return sorted(profiles)[:20]


# ─── Curses TUI Main Page ──────────────────────────────────────────


def run_main_page():
    """Launch the interactive TUI entry point."""
    profiles = _find_profiles()

    if not profiles:
        # No profiles — fall back to text help
        show_help()
        return

    try:
        curses.wrapper(_main_tui, profiles)
    except curses.error:
        # Terminal too small or no curses support — fall back to simple mode
        _run_simple_mode(profiles)


def _main_tui(stdscr, profiles: list[str]):
    """Curses-based main page."""
    curses.curs_set(0)
    curses.use_default_colors()

    # Init color pairs
    if curses.has_colors():
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # banner
        curses.init_pair(2, curses.COLOR_GREEN, -1)     # selected
        curses.init_pair(3, curses.COLOR_WHITE, -1)     # normal
        curses.init_pair(4, curses.COLOR_YELLOW, -1)    # header
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)  # highlight bar

    # State
    phase = "profile"  # "profile" or "action"
    profile_idx = 0
    action_idx = 0
    selected_profile = None

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── Banner ──
        _draw_banner(stdscr, w)
        row = 9

        if phase == "profile":
            _draw_profile_picker(stdscr, profiles, profile_idx, row, w)
        else:
            _draw_action_picker(stdscr, selected_profile, action_idx, row, w)

        # Footer
        footer_row = min(row + len(profiles) + len(ACTIONS) + 8, h - 1)
        if footer_row < h:
            hint = "  ↑↓ Navigate  ⏎ Select  q Quit"
            if phase == "action":
                hint = "  ↑↓ Navigate  ⏎ Select  ← Back  q Quit"
            stdscr.addnstr(footer_row, 0, hint, w - 1, curses.color_pair(3))

        stdscr.refresh()

        # ── Input ──
        key = stdscr.getch()

        if key == ord("q") or key == 27:  # q or ESC
            if phase == "action":
                phase = "profile"
                continue
            return

        if key == curses.KEY_UP:
            if phase == "profile":
                profile_idx = max(0, profile_idx - 1)
            else:
                action_idx = max(0, action_idx - 1)

        elif key == curses.KEY_DOWN:
            if phase == "profile":
                profile_idx = min(len(profiles) - 1, profile_idx + 1)
            else:
                action_idx = min(len(ACTIONS) - 1, action_idx + 1)

        elif key == curses.KEY_LEFT and phase == "action":
            phase = "profile"

        elif key in (curses.KEY_ENTER, 10, 13):
            if phase == "profile":
                selected_profile = profiles[profile_idx]
                if len(profiles) == 1:
                    # Skip straight to action picker for single profile
                    pass
                phase = "action"
                action_idx = 0
            else:
                # Launch the selected action
                cmd, _, _ = ACTIONS[action_idx]
                curses.endwin()
                _launch_action(cmd, selected_profile)
                return


def _draw_banner(stdscr, w):
    """Draw the ASCII banner at the top."""
    lines = [
        "  ┌─────────────────────────────────────────────┐",
        "  │              🔬  nsys-ai                     │",
        "  │   AI-powered GPU profile analysis            │",
        "  │                                              │",
        "  │   Navigate timelines · Diagnose bottlenecks  │",
        "  │   Explore NVTX trees · Run analysis skills   │",
        "  └─────────────────────────────────────────────┘",
    ]
    for i, line in enumerate(lines):
        try:
            stdscr.addnstr(i + 1, 0, line, w - 1, curses.color_pair(1))
        except curses.error:
            pass


def _draw_profile_picker(stdscr, profiles, selected, start_row, w):
    """Draw the profile selection list."""
    try:
        stdscr.addnstr(start_row, 2, "Select a profile:", w - 3,
                       curses.color_pair(4) | curses.A_BOLD)
    except curses.error:
        pass

    for i, p in enumerate(profiles):
        row = start_row + 2 + i
        if row >= curses.LINES - 2:
            break

        size_kb = os.path.getsize(p) // 1024 if os.path.exists(p) else 0
        label = f"  {p}  ({size_kb:,} KB)"

        if i == selected:
            attr = curses.color_pair(5) | curses.A_BOLD
            label = f"▸ {p}  ({size_kb:,} KB)"
        else:
            attr = curses.color_pair(3)
            label = f"  {p}  ({size_kb:,} KB)"

        try:
            stdscr.addnstr(row, 2, label.ljust(w - 4), w - 3, attr)
        except curses.error:
            pass


def _draw_action_picker(stdscr, profile, selected, start_row, w):
    """Draw the action selection list."""
    try:
        basename = os.path.basename(profile)
        stdscr.addnstr(start_row, 2, f"Profile: {basename}", w - 3,
                       curses.color_pair(4) | curses.A_BOLD)
        stdscr.addnstr(start_row + 1, 2, "What would you like to do?", w - 3,
                       curses.color_pair(3))
    except curses.error:
        pass

    for i, (cmd, icon, desc) in enumerate(ACTIONS):
        row = start_row + 3 + i
        if row >= curses.LINES - 2:
            break

        if i == selected:
            attr = curses.color_pair(5) | curses.A_BOLD
            label = f"▸ {icon} {desc}"
        else:
            attr = curses.color_pair(3)
            label = f"  {icon} {desc}"

        try:
            stdscr.addnstr(row, 2, label.ljust(w - 4), w - 3, attr)
        except curses.error:
            pass


# ─── Fallback simple mode (no curses) ──────────────────────────────


def _run_simple_mode(profiles: list[str]):
    """Text-based fallback when curses isn't available."""
    print(BANNER)
    print(f"  Found {len(profiles)} profile(s):\n")
    for i, p in enumerate(profiles, 1):
        size_kb = os.path.getsize(p) // 1024
        print(f"    [{i}] {p}  ({size_kb:,} KB)")
    print(f"\n    [q] Quit\n")

    try:
        choice = input("  Select profile: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice.lower() == "q":
        return
    try:
        profile = profiles[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return

    print(f"\n  Profile: {profile}\n  Actions:\n")
    for i, (cmd, icon, desc) in enumerate(ACTIONS, 1):
        print(f"    [{i}] {icon} {desc}")
    print(f"\n    [q] Quit\n")

    try:
        choice = input("  Select action: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice.lower() == "q":
        return
    try:
        cmd, _, _ = ACTIONS[int(choice) - 1]
        _launch_action(cmd, profile)
    except (ValueError, IndexError):
        print("  Invalid selection.")


# ─── Help (nsys-ai help) ───────────────────────────────────────────


def show_help():
    """Print getting-started guide and command reference."""
    print(BANNER)
    print("  Commands:")
    print("  ─────────────────────────────────────────────────────────")
    print("    nsys-ai                       Interactive main page")
    print("    nsys-ai help                  This help text")
    print()
    print("  Analysis:")
    print("    nsys-ai info    <profile>                Profile metadata & GPUs")
    print("    nsys-ai summary <profile> [--gpu N]      Kernel stats & commentary")
    print("    nsys-ai timeline <profile> --gpu N --trim S E   Timeline TUI")
    print("    nsys-ai tui     <profile> --gpu N --trim S E   Tree TUI")
    print()
    print("  Skills & Agent:")
    print("    nsys-ai skill list                       List analysis skills")
    print("    nsys-ai skill run <name> <profile>       Run a specific skill")
    print("    nsys-ai agent analyze <profile>           Full auto-analysis")
    print("    nsys-ai agent ask <profile> \"question\"   Ask about a profile")
    print()
    print("  Export:")
    print("    nsys-ai export     <profile> -o DIR       Perfetto JSON traces")
    print("    nsys-ai export-csv <profile> --gpu N       CSV export")
    print("    nsys-ai viewer     <profile> --gpu N       HTML report")
    print("    nsys-ai web        <profile> --gpu N       Browser UI")
    print()
    print("  Getting Started:")
    print("    1. Profile:  nsys profile -o report python train.py")
    print("    2. Export:   nsys export --type sqlite report.nsys-rep")
    print("    3. Explore:  nsys-ai")
    print()


# ─── Action launcher ───────────────────────────────────────────────


def _launch_action(cmd: str, profile: str):
    """Launch the selected action for a profile."""
    from . import profile as _profile

    if cmd == "skill":
        from .skills.registry import all_skills, run_skill
        import sqlite3

        skills = all_skills()
        print(f"\n  Available skills:\n")
        for i, s in enumerate(skills, 1):
            print(f"    [{i}] {s.name:<25s}  {s.description[:55]}")
        print(f"\n    [q] Back\n")

        try:
            choice = input("  Select skill: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice.lower() == "q":
            return
        try:
            conn = sqlite3.connect(profile)
            print()
            print(run_skill(skills[int(choice) - 1].name, conn))
            conn.close()
        except (ValueError, IndexError, Exception) as e:
            print(f"  Error: {e}")
        return

    if cmd == "agent":
        from .agent.loop import Agent
        agent = Agent(profile)
        try:
            print()
            print(agent.analyze())
        finally:
            agent.close()
        return

    # Commands needing GPU + trim
    prof = _profile.open(profile)
    meta = prof.meta

    # Auto-select GPU
    gpu = meta.devices[0] if meta.devices else 0
    trim = meta.time_range  # full range

    if cmd == "info":
        print(f"\n  Profile: {profile}")
        print(f"    GPUs: {meta.devices}")
        print(f"    Kernels: {meta.kernel_count}  |  NVTX: {meta.nvtx_count}")
        print(f"    Time: {meta.time_range[0]/1e9:.3f}s – {meta.time_range[1]/1e9:.3f}s")
        for dev, info in meta.gpu_info.items():
            print(f"    GPU {dev}: {info.name} | SMs={info.sm_count} | "
                  f"Mem={info.memory_bytes/1e9:.0f}GB | Kernels={info.kernel_count}")
        prof.close()

    elif cmd == "summary":
        from .summary import gpu_summary, format_text, auto_commentary
        s = gpu_summary(prof, gpu, trim)
        print()
        print(format_text(s))
        print()
        print(auto_commentary(s))
        prof.close()

    elif cmd == "timeline":
        prof.close()
        from .tui_timeline import run_timeline
        run_timeline(profile, gpu, trim)

    elif cmd == "tui":
        prof.close()
        from .tui import run_tui
        run_tui(profile, gpu, trim)

    elif cmd == "web":
        from .web import serve
        serve(prof, gpu, trim)
