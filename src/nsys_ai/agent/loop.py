"""
loop.py — Core agent analysis loop.

The Agent takes a profile, selects relevant skills, executes them,
and produces a structured analysis report. Works without LLM by default
(keyword-based skill selection + template reporting). With the [agent]
extra installed, can delegate to an LLM for natural language analysis.
"""

import logging
import shlex
import sqlite3

from ..exceptions import NsysAiError, ProfileNotFoundError
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
        "nccl": [
            "nccl_breakdown",
            "nccl_communicator_analysis",
            "overlap_breakdown",
            "kernel_overlap_matrix",
            "nccl_anomaly",
        ],
        "allreduce": ["nccl_breakdown", "nccl_communicator_analysis", "nccl_anomaly"],
        "collective": ["nccl_breakdown", "nccl_communicator_analysis", "nccl_anomaly"],
        "distributed": [
            "nccl_breakdown",
            "nccl_communicator_analysis",
            "overlap_breakdown",
            "kernel_overlap_matrix",
            "nccl_anomaly",
        ],
        "multi-gpu": [
            "nccl_breakdown",
            "nccl_communicator_analysis",
            "overlap_breakdown",
            "kernel_overlap_matrix",
        ],
        "communicator": ["nccl_communicator_analysis", "nccl_breakdown"],
        "rank": ["nccl_communicator_analysis", "nccl_breakdown"],
        "tensor parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
        "pipeline parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
        "data parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
        "anomaly": ["nccl_anomaly"],
        "outlier": ["nccl_anomaly"],
        "overlap": ["overlap_breakdown", "kernel_overlap_matrix"],
        "matrix": ["kernel_overlap_matrix"],
        "contention": ["kernel_overlap_matrix", "stream_concurrency"],
        "hidden": ["kernel_overlap_matrix", "overlap_breakdown"],
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
        except ProfileNotFoundError:
            # A missing file has nothing to fall back to — fail cleanly rather
            # than letting sqlite3.connect below create an empty stub.
            raise
        except (NsysAiError, sqlite3.Error, ValueError) as e:
            import sqlite3 as _sqlite3

            log.warning(
                "Could not open as Nsight profile (skills may be limited): %s",
                e,
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
        1. top_kernels
        2. gpu_idle_gaps
        3. memory_transfers
        4. memory_bandwidth
        5. nccl_breakdown
        6. nccl_communicator_analysis
        7. nccl_anomaly
        8. kernel_launch_overhead
        9. kernel_launch_pattern
        10. stream_concurrency
        11. overlap_breakdown
        12. kernel_overlap_matrix
        13. iteration_timing
        14. nvtx_layer_breakdown

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
            "nccl_communicator_analysis",
            "nccl_anomaly",
            "kernel_launch_overhead",
            "kernel_launch_pattern",
            "stream_concurrency",
            "overlap_breakdown",
            "kernel_overlap_matrix",
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

        evidence = {}

        # Stage 1: Triage (Unconditional root_cause_matcher)
        triage_skill = "root_cause_matcher"
        try:
            skill = get_skill(triage_skill)
            if skill:
                rows = skill.execute(self.conn, **self._trim_kwargs)
                evidence[triage_skill] = rows
        except Exception as e:
            log.debug("Triage skill '%s' failed: %s", triage_skill, e, exc_info=True)

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
            except Exception as e:
                log.debug("Skill '%s' failed: %s", skill_name, e, exc_info=True)

        # Try LLM synthesis with combined structured evidence
        llm_answer = None
        if has_llm:
            llm_answer = self._try_llm_synthesis(question, evidence)

        answer = self._format_evidence_first_answer(
            question,
            evidence,
            selected_skills=[triage_skill, *selected],
            llm_answer=llm_answer,
        )
        return answer

    def run_skill(self, skill_name: str, **kwargs) -> str:
        """Run a specific skill by name."""
        return run_skill(skill_name, self.conn, **kwargs)

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

    def _format_evidence_first_answer(
        self,
        question: str,
        evidence: dict[str, list[dict]],
        selected_skills: list[str],
        llm_answer: str | None = None,
    ) -> str:
        """Build the fixed answer shape required by issue #205."""
        selected_skills = list(dict.fromkeys(skill for skill in selected_skills if skill))
        diagnosis_row = self._first_actionable_row(evidence.get("root_cause_matcher", []))
        diagnosis = self._primary_diagnosis(question, evidence, diagnosis_row)
        evidence_lines = self._evidence_lines(evidence)
        confidence = self._confidence_label(evidence, diagnosis_row)
        action = self._recommended_action(diagnosis_row)
        verify_skill = self._choose_verify_skill(evidence, selected_skills)
        verify_command = self._verify_command(verify_skill)

        if llm_answer:
            summary = (
                "Ran targeted skills and generated a grounded answer from their structured "
                "outputs; model synthesis was available, but the final answer is constrained "
                "to the evidence-first template."
            )
        else:
            ran = ", ".join(skill for skill in selected_skills if skill)
            if ran:
                summary = (
                    f"Ran {ran} against the profile and summarized the strongest supported "
                    "signal in a verification-friendly format."
                )
            else:
                summary = (
                    "No skill returned usable evidence, so the answer is limited to a "
                    "verification fallback."
                )

        lines = [
            "## Summary",
            summary,
            "",
            "## Primary Diagnosis",
            diagnosis,
            "",
            "## Evidence",
        ]
        if evidence_lines:
            lines.extend(evidence_lines)
        else:
            lines.append(
                "- source_skill=none; metric=none; window=full profile; "
                "scope=profile; evidence=no skill returned usable rows"
            )
        lines.extend(
            [
                "",
                "## Confidence",
                confidence,
                "",
                "## Recommended Action",
                action,
                "",
                "## Verify",
            ]
        )
        if verify_command:
            lines.append(f"`{verify_command}`")
        else:
            lines.append(
                "Could not build a runnable verification command because no skill "
                "produced evidence. Inspect available skills with:"
            )
            lines.append("`nsys-ai skill list`")
        return "\n".join(lines)

    def _first_actionable_row(self, rows: list[dict]) -> dict | None:
        for row in rows:
            pattern = str(row.get("pattern", ""))
            if pattern and pattern != "No Known Anti-Patterns Detected":
                return row
        return None

    def _primary_diagnosis(
        self,
        question: str,
        evidence: dict[str, list[dict]],
        diagnosis_row: dict | None,
    ) -> str:
        if diagnosis_row:
            pattern = diagnosis_row.get("pattern") or diagnosis_row.get("label")
            if pattern:
                return str(pattern)
        for skill_name, rows in evidence.items():
            if rows:
                row = rows[0]
                label = row.get("label") or row.get("name") or row.get("kernel_name")
                if label:
                    return f"{label} ({skill_name})"
        return f"No specific diagnosis could be grounded for: {question}"

    def _recommended_action(self, diagnosis_row: dict | None) -> str:
        if diagnosis_row:
            rec = diagnosis_row.get("recommendation") or diagnosis_row.get("action")
            if rec:
                return str(rec)
        return (
            "Re-run the verify command, inspect the cited metrics and window, then collect "
            "a narrower profile with NVTX ranges if the evidence is too broad."
        )

    def _confidence_label(self, evidence: dict[str, list[dict]], diagnosis_row: dict | None) -> str:
        row_count = sum(len(rows) for rows in evidence.values())
        if diagnosis_row and row_count:
            return "0.75 (medium-high): at least one root-cause matcher finding is backed by skill output."
        if row_count:
            return "0.60 (medium): skill output exists, but no root-cause matcher finding dominated."
        return "0.20 (low): no skill returned usable evidence."

    def _evidence_lines(self, evidence: dict[str, list[dict]]) -> list[str]:
        lines: list[str] = []
        for skill_name, rows in evidence.items():
            for row in rows[:2]:
                if not isinstance(row, dict):
                    continue
                if row.get("_summary") and len(rows) > 1:
                    continue
                metric = self._metric_fragment(row)
                window = self._window_fragment(row)
                scope = self._scope_fragment(row)
                evidence_text = str(row.get("evidence") or row.get("note") or "").strip()
                suffix = f"; evidence={evidence_text}" if evidence_text else ""
                lines.append(
                    f"- source_skill={skill_name}; metric={metric}; "
                    f"window={window}; scope={scope}{suffix}"
                )
                if len(lines) >= 5:
                    return lines
        return lines

    def _metric_fragment(self, row: dict) -> str:
        priority = (
            "pattern",
            "label",
            "name",
            "kernel_name",
            "severity",
            "total_ms",
            "duration_ms",
            "gap_ms",
            "gap_ns",
            "idle_pct",
            "total_idle_ms",
            "overlap_pct",
            "nccl_only_ms",
            "compute_only_ms",
            "count",
        )
        parts = []
        for key in priority:
            if key in row and row[key] not in (None, ""):
                parts.append(f"{key}={self._compact_value(row[key])}")
            if len(parts) >= 3:
                break
        return ", ".join(parts) if parts else "row_present=true"

    def _compact_value(self, value) -> str:
        text = str(value)
        return text if len(text) <= 120 else text[:117] + "..."

    def _window_fragment(self, row: dict) -> str:
        start = row.get("start_ns", row.get("gpu_start_ns"))
        end = row.get("end_ns", row.get("gpu_end_ns"))
        if start is not None and end is not None:
            return f"{start}-{end}ns"
        if row.get("start_ms") is not None and row.get("end_ms") is not None:
            return f"{row['start_ms']}-{row['end_ms']}ms"
        trim_start = self._trim_kwargs.get("trim_start_ns")
        trim_end = self._trim_kwargs.get("trim_end_ns")
        if trim_start is not None and trim_end is not None:
            return f"{trim_start}-{trim_end}ns"
        return "full profile"

    def _scope_fragment(self, row: dict) -> str:
        parts = []
        for key in ("gpu_id", "device_id", "device", "rank", "stream_id", "communicator_hex"):
            if key in row and row[key] not in (None, ""):
                parts.append(f"{key}={row[key]}")
        return ", ".join(parts) if parts else "profile"

    def _choose_verify_skill(
        self,
        evidence: dict[str, list[dict]],
        selected_skills: list[str],
    ) -> str | None:
        for skill_name in selected_skills:
            rows = evidence.get(skill_name)
            if rows:
                return skill_name
        for skill_name, rows in evidence.items():
            if rows:
                return skill_name
        return None

    def _verify_command(self, skill_name: str | None) -> str | None:
        if not skill_name:
            return None
        cmd = [
            "nsys-ai",
            "skill",
            "run",
            skill_name,
            self.profile_path,
            "--format",
            "json",
        ]
        trim_start = self._trim_kwargs.get("trim_start_ns")
        trim_end = self._trim_kwargs.get("trim_end_ns")
        if trim_start is not None and trim_end is not None:
            cmd.extend(["--trim", f"{trim_start / 1e9:g}", f"{trim_end / 1e9:g}"])
        return " ".join(shlex.quote(str(part)) for part in cmd)

    def _try_llm_synthesis(self, question: str, evidence: dict[str, list[dict]]) -> str | None:
        """Try to use an LLM to synthesize an answer from structured evidence.

        Args:
            question: The question to answer.
            evidence: Dict mapping skill names to their JSON-serializable results.

        Returns None if no LLM available.
        """
        import json
        import os

        def _build_system_with_trace_context() -> str:
            try:
                from .persona import build_system_prompt

                system_str = build_system_prompt()
                fp_str = ""
                if getattr(self, "profile", None) and getattr(self.profile, "fingerprint", None):
                    fp_str = self.profile.fingerprint.to_prompt_string()

                return (
                    (
                        f"{system_str}\n\n"
                        f"--- TRACE CONTEXT ---\n{fp_str}\n---------------------\n"
                        "Apply framework-specific knowledge when diagnosing bottlenecks."
                    )
                    if fp_str
                    else system_str
                )
            except Exception:
                log.debug("Failed to load persona prompt", exc_info=True)
                return "You are an expert GPU profiling assistant."

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
                system = _build_system_with_trace_context()

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
            system = _build_system_with_trace_context()

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return message.content[0].text
        except Exception as e:
            log.debug("LLM synthesis (anthropic) failed: %s", e, exc_info=True)
            return f"(LLM synthesis failed: {e})"
