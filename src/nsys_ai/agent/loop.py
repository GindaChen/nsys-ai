"""
loop.py — Core agent analysis loop.

The Agent takes a profile, selects relevant skills, executes them,
and produces a structured analysis report. Works without LLM by default
(keyword-based skill selection + template reporting). With the [agent]
extra installed, can delegate to an LLM for natural language analysis.
"""


import logging
import sqlite3

from ..exceptions import NsysAiError
from ..profile import Profile
from ..skills.registry import get_skill, run_skill

log = logging.getLogger(__name__)


class Agent:
    """GPU profile analysis agent.

    Usage:
        agent = Agent("profile.sqlite")
        report = agent.analyze()         # auto-report
        answer = agent.ask("why slow?")  # targeted question
    """

    # Keywords → skills mapping for non-LLM skill selection
    _KEYWORD_MAP = {
        "kernel": ["top_kernels", "kernel_launch_overhead"],
        "hotspot": ["top_kernels"],
        "slow": ["top_kernels", "gpu_idle_gaps"],
        "bubble": ["gpu_idle_gaps"],
        "idle": ["gpu_idle_gaps"],
        "gap": ["gpu_idle_gaps"],
        "stall": ["gpu_idle_gaps", "nccl_anomaly"],
        "memory": ["memory_transfers", "memory_bandwidth"],
        "transfer": ["memory_transfers", "memory_bandwidth"],
        "h2d": ["memory_transfers", "memory_bandwidth"],
        "copy": ["memory_transfers", "memory_bandwidth"],
        "bandwidth": ["memory_bandwidth"],
        "nccl": ["nccl_breakdown", "overlap_breakdown", "nccl_anomaly"],
        "allreduce": ["nccl_breakdown", "nccl_anomaly"],
        "collective": ["nccl_breakdown", "nccl_anomaly"],
        "distributed": ["nccl_breakdown", "overlap_breakdown", "nccl_anomaly"],
        "multi-gpu": ["nccl_breakdown", "overlap_breakdown"],
        "anomaly": ["nccl_anomaly"],
        "outlier": ["nccl_anomaly"],
        "overlap": ["overlap_breakdown"],
        "nvtx": ["nvtx_kernel_map", "nvtx_layer_breakdown"],
        "source": ["nvtx_kernel_map"],
        "attribution": ["nvtx_kernel_map"],
        "mapping": ["nvtx_kernel_map"],
        "layer": ["nvtx_layer_breakdown"],
        "launch": ["kernel_launch_overhead", "kernel_launch_pattern"],
        "overhead": ["kernel_launch_overhead"],
        "dispatch": ["kernel_launch_pattern"],
        "pattern": ["kernel_launch_pattern"],
        "burst": ["kernel_launch_pattern"],
        "stream": ["stream_concurrency"],
        "concurrency": ["stream_concurrency"],
        "parallel": ["stream_concurrency"],
        "serial": ["stream_concurrency"],
        "cpu": ["thread_utilization", "cpu_gpu_pipeline"],
        "thread": ["thread_utilization"],
        "utilization": ["thread_utilization", "stream_concurrency"],
        "pipeline": ["cpu_gpu_pipeline"],
        "starvation": ["cpu_gpu_pipeline"],
        "queue": ["cpu_gpu_pipeline"],
        "schema": ["schema_inspect"],
        "table": ["schema_inspect"],
        "mfu": ["region_mfu", "theoretical_flops"],
        "flops": ["theoretical_flops"],
        "efficiency": ["region_mfu"],
        "iteration": ["iteration_timing"],
        "iter": ["iteration_timing"],
        "training": ["iteration_timing"],
        "step": ["iteration_timing"],
        "diagnosis": ["root_cause_matcher"],
        "root-cause": ["root_cause_matcher"],
        "why": ["root_cause_matcher"],
        "speedup": ["speedup_estimator"],
        "estimate": ["speedup_estimator"],
        "projection": ["speedup_estimator"],
    }

    def __init__(self, profile_path: str, trim_ns: tuple[int, int] | None = None):
        self.profile_path = profile_path
        self._trim_kwargs: dict = {}
        if trim_ns:
            self._trim_kwargs["trim_start_ns"] = trim_ns[0]
            self._trim_kwargs["trim_end_ns"] = trim_ns[1]
        try:
            self.profile = Profile(profile_path)
        except (NsysAiError, sqlite3.Error, ValueError) as e:
            import sqlite3 as _sqlite3
            log.warning(
                "Could not open as Nsight profile (skills may be limited): %s", e,
            )
            # Fallback: open as a raw SQLite connection so the agent can still
            # run generic SQL queries even if schema detection fails.
            self.profile = None  # type: ignore[assignment]
            self.conn = _sqlite3.connect(profile_path, check_same_thread=False)
            self.conn.row_factory = _sqlite3.Row
            return
        self.conn = self.profile.db if self.profile.db is not None else self.profile.conn

    def close(self):
        if self.profile is not None:
            self.profile.close()
        elif hasattr(self, "conn"):
            self.conn.close()

    def analyze(self) -> str:
        """Run a full auto-analysis of the profile.

        Executes the core skills in order:
        1. top_kernels — identify hotspots
        2. gpu_idle_gaps — find pipeline bubbles
        3. memory_transfers — check data movement
        4. nccl_breakdown — check collective overhead (if present)
        5. kernel_launch_overhead — check dispatch latency
        6. overlap_breakdown — compute/NCCL overlap analysis
        7. iteration_timing — per-iteration timing
        8. nvtx_layer_breakdown — per-NVTX-region GPU time

        Returns:
            Formatted multi-section report with optional AI synthesis.
        """
        sections = []
        sections.append("═══ nsys-ai Auto-Analysis Report ═══\n")

        # Structured evidence for LLM (JSON-serializable)
        evidence = {}

        # Always run these core skills
        core_skills = [
            "top_kernels",
            "gpu_idle_gaps",
            "memory_transfers",
            "memory_bandwidth",
            "nccl_breakdown",
            "nccl_anomaly",
            "kernel_launch_overhead",
            "kernel_launch_pattern",
            "stream_concurrency",
            "overlap_breakdown",
            "iteration_timing",
            "nvtx_layer_breakdown",
        ]

        for skill_name in core_skills:
            try:
                skill = get_skill(skill_name)
                if skill is None:
                    continue
                rows = skill.execute(self.conn, **self._trim_kwargs)
                evidence[skill_name] = rows
                text = skill.format_rows(rows)
                sections.append(text)
                sections.append("")
            except Exception as e:
                log.debug("Skill '%s' failed: %s", skill_name, e, exc_info=True)
                sections.append(f"({skill_name}: skipped — {e})\n")

        # LLM synthesis with structured JSON evidence
        llm_answer = self._try_llm_synthesis(
            "Provide a comprehensive GPU performance analysis based on the profile data.",
            evidence,
        )
        if llm_answer:
            sections.append("\n── AI Analysis ──")
            sections.append(llm_answer)

        sections.append("═══ End of Report ═══")
        return "\n".join(sections)

    def ask(self, question: str) -> str:
        """Answer a natural language question about the profile.

        Uses a two-stage process:
        1. Triage: Runs root_cause_matcher to gather baseline signals.
        2. Deep Dive: Uses an LLM to select targeted skills based on the triage signals,
           executes them, and synthesizes a final response. If no LLM, falls back to keywords.
        """
        # Use shared chat configuration to determine if an LLM is available
        try:
            from ..chat_config import _get_model_and_key

            model, api_key = _get_model_and_key()
        except Exception:
            log.debug("LLM model/key resolution failed", exc_info=True)
            model, api_key = None, None
        has_llm = bool(model and api_key)

        sections = [f"Question: {question}\n"]
        evidence = {}

        # Stage 1: Triage (Unconditional root_cause_matcher)
        triage_skill = "root_cause_matcher"
        try:
            skill = get_skill(triage_skill)
            if skill:
                rows = skill.execute(self.conn, **self._trim_kwargs)
                evidence[triage_skill] = rows
                sections.append("── Phase 1: Triage (Root Cause Matcher) ──")
                sections.append(skill.format_rows(rows))
                sections.append("")
        except Exception as e:
            log.debug("Triage skill '%s' failed: %s", triage_skill, e, exc_info=True)
            sections.append(f"({triage_skill}: skipped — {e})\n")

        # Select Deep Dive Skills
        if has_llm:
            selected = self._try_llm_triage(question, evidence.get(triage_skill, []))
            # Filter out triage skill and drop empty entries
            selected = [s for s in selected if s and s != triage_skill]
            # Fallback if LLM returned nothing usable
            if not selected:
                selected = self._select_skills(question)
            if not selected:
                selected = ["top_kernels", "gpu_idle_gaps"]
            sections.append(f"── Phase 2: AI Triage selected skills: {', '.join(selected)} ──\n")
        else:
            selected = self._select_skills(question)
            if not selected:
                selected = ["top_kernels", "gpu_idle_gaps"]

        # Stage 2: Deep Dive (Execute selected skills)
        for skill_name in selected:
            if skill_name == triage_skill:
                continue
            try:
                skill = get_skill(skill_name)
                if skill is None:
                    continue
                rows = skill.execute(self.conn, **self._trim_kwargs)
                evidence[skill_name] = rows
                text = skill.format_rows(rows)
                sections.append(text)
                sections.append("")
            except Exception as e:
                log.debug("Skill '%s' failed: %s", skill_name, e, exc_info=True)
                sections.append(f"({skill_name}: skipped — {e})\n")

        # Try LLM synthesis with combined structured evidence
        if has_llm:
            llm_answer = self._try_llm_synthesis(question, evidence)
            if llm_answer:
                sections.append("── Phase 3: AI Final Analysis ──")
                sections.append(llm_answer)

        return "\n".join(sections)

    def run_skill(self, name: str, **kwargs) -> str:
        """Run a specific skill by name."""
        return run_skill(name, self.conn, **kwargs)

    def _try_llm_triage(self, question: str, triage_results: list[dict]) -> list[str]:
        """Use LLM to select the next set of skills based on the triage findings."""
        import json

        from ..skills.registry import list_skills

        available_skills = list_skills()
        triage_json = json.dumps(triage_results, indent=2, default=str)

        prompt = (
            f"You are a performance profiling expert. The user asked: '{question}'.\n"
            f"We ran a triage check (`root_cause_matcher`) and found these signals:\n"
            f"```json\n{triage_json}\n```\n\n"
            f"Available skills you can run to investigate further: {', '.join(available_skills)}\n\n"
            f"Based on the user's question and the triage findings, select up to 4 skill names "
            f"to run in a deep-dive investigation. Respond ONLY with a comma-separated list of skill names, "
            f"like 'top_kernels, gpu_idle_gaps'. Do not provide any other text."
        )

        try:
            import litellm

            from ..chat_config import _get_model_and_key

            model, _ = _get_model_and_key()

            if model:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                )
                text_response = resp.choices[0].message.content.strip()
                # Parse returned text into a list of skills
                selected = []
                for s in text_response.split(","):
                    s = s.strip()
                    # Strip any markdown backticks or quotes that the LLM might have included
                    s = s.replace("`", "").replace("'", "").replace('"', "")
                    if s in available_skills:
                        selected.append(s)
                return selected[:4]
        except Exception:
            log.debug("LLM triage failed, falling back to keywords", exc_info=True)
            pass

        # Fallback to keywords if LLM fails
        return self._select_skills(question)

    def _select_skills(self, question: str) -> list[str]:
        """Select skills relevant to a question using keyword matching."""
        q_lower = question.lower()
        selected = set()
        for keyword, skill_names in self._KEYWORD_MAP.items():
            if keyword in q_lower:
                selected.update(skill_names)
        return sorted(selected)

    def _try_llm_synthesis(self, question: str, evidence: dict[str, list[dict]]) -> str | None:
        """Try to use an LLM to synthesize an answer from structured evidence.

        Args:
            question: The question to answer.
            evidence: Dict mapping skill names to their JSON-serializable results.

        Returns None if no LLM available.
        """
        import json
        import os

        evidence_json = json.dumps(evidence, indent=2, default=str)
        user_msg = (
            f"Profile analysis data (structured JSON):\n"
            f"```json\n{evidence_json}\n```\n\n"
            f"Based on this data, answer the following question:\n{question}"
        )

        # Try litellm first (supports Gemini, OpenAI, Anthropic, etc.)
        try:
            import litellm

            # Pick best available model based on API keys
            model = None
            if os.environ.get("GEMINI_API_KEY"):
                model = "gemini/gemini-2.5-flash"
            elif os.environ.get("OPENAI_API_KEY"):
                model = "gpt-4o-mini"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                model = "claude-sonnet-4-20250514"

            if model:
                try:
                    from .persona import build_system_prompt

                    system = build_system_prompt()
                except Exception:
                    log.debug("Failed to load persona prompt", exc_info=True)
                    system = "You are an expert GPU profiling assistant."

                resp = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=2048,
                )
                return resp.choices[0].message.content
        except ImportError:
            pass
        except Exception as e:
            log.debug("LLM synthesis (litellm) failed: %s", e, exc_info=True)
            return f"(LLM synthesis failed: {e})"

        # Fallback: direct Anthropic SDK (legacy path)
        try:
            import anthropic
        except ImportError:
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        try:
            from .persona import build_system_prompt

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=build_system_prompt(),
                messages=[{"role": "user", "content": user_msg}],
            )
            return message.content[0].text
        except Exception as e:
            log.debug("LLM synthesis (anthropic) failed: %s", e, exc_info=True)
            return f"(LLM synthesis failed: {e})"
