"""
loop.py — Core agent analysis loop.

The Agent takes a profile, selects relevant skills, executes them,
and produces a structured analysis report. Works without LLM by default
(keyword-based skill selection + template reporting). With the [agent]
extra installed, can delegate to an LLM for natural language analysis.
"""

import hashlib
import logging
import shlex
import sqlite3
from dataclasses import dataclass

from ..annotation import Diagnostic
from ..exceptions import NsysAiError, ProfileNotFoundError
from ..profile import Profile
from ..skills.registry import get_skill, run_skill

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Structured outcome of an ``ask()`` / ``analyze()`` run.

    ``text`` is the human-readable Markdown answer; ``diagnostic`` is the
    same conclusion in the v0.1 :class:`~nsys_ai.annotation.Diagnostic`
    schema. Both are rendered from one structured result — never by parsing
    the Markdown back — so the terminal output and ``diagnostics.json``
    cannot drift apart.
    """

    text: str
    diagnostic: Diagnostic


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
        # Lazily resolved by _profile_id(); cached so repeated
        # ask_result()/analyze_result() calls hash the profile only once.
        self._cached_profile_id: str | None = None
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

    #: Fixed question recorded for analyze-mode diagnostics. Kept as a
    #: constant so the deterministic diagnostic id is stable across runs.
    _ANALYZE_QUESTION = (
        "Provide a comprehensive GPU performance analysis based on the profile data."
    )

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
        return self.analyze_result().text

    def analyze_result(self) -> AgentResult:
        """Run a full auto-analysis and return text plus a structured diagnostic.

        Same skill execution as :meth:`analyze`; the diagnostic is derived
        from the evidence rows collected here — no skill is re-run and no
        ``EvidenceBuilder`` pass is needed.
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

        # Request the same concise summary that is persisted in Diagnostic.
        # The structured diagnosis block below is the only human rendering of
        # that synthesis, so terminal and JSON output cannot diverge.
        llm_summary = self._try_llm_synthesis(
            self._ANALYZE_QUESTION,
            evidence,
            summary_only=True,
        )

        diagnostic = self._build_diagnostic(
            mode="analyze",
            question=self._ANALYZE_QUESTION,
            evidence=evidence,
            selected_skills=core_skills,
            llm_summary=llm_summary,
        )
        sections.append("")
        sections.append("── Structured Diagnosis ──")
        sections.append(self._render_answer_text(diagnostic))
        sections.append("═══ End of Report ═══")
        return AgentResult(text="\n".join(sections), diagnostic=diagnostic)

    def ask(self, question: str) -> str:
        """Answer a natural language question about the profile.

        Uses a two-stage process:
        1. Triage: Runs root_cause_matcher to gather baseline signals.
        2. Deep Dive: Uses an LLM to select targeted skills based on the triage signals,
           executes them, and synthesizes the Summary. If no LLM, falls back to keywords
           and a deterministic Summary. The remaining answer sections are always built
           deterministically from skill evidence.
        """
        return self.ask_result(question).text

    def ask_result(self, question: str) -> AgentResult:
        """Answer a question and return the text plus a structured diagnostic.

        The human-readable sections and the ``diagnostics.json`` fields are
        rendered from the same :class:`Diagnostic` (see
        :meth:`_build_diagnostic`), so the two outputs always agree.
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

        # Ask the LLM for the summary that will be placed into the deterministic
        # evidence-first answer shape below.
        llm_summary = None
        if has_llm:
            llm_summary = self._try_llm_synthesis(question, evidence, summary_only=True)

        diagnostic = self._build_diagnostic(
            mode="ask",
            question=question,
            evidence=evidence,
            selected_skills=[triage_skill, *selected],
            llm_summary=llm_summary,
        )
        return AgentResult(
            text=self._render_answer_text(diagnostic),
            diagnostic=diagnostic,
        )

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

    #: Fallback verification command used when no skill produced usable
    #: rows. It is still a runnable ``nsys-ai`` command (the user lists the
    #: available skills and picks one manually), so the JSON field never
    #: carries prose.
    _VERIFY_FALLBACK_COMMAND = "nsys-ai skill list"

    #: Upper bound on findings embedded in ``Diagnostic.primary_findings``.
    #: Keeps ``diagnostics.json`` focused on the primary evidence even when
    #: a skill's converter would emit hundreds of findings.
    _MAX_PRIMARY_FINDINGS = 10

    def _build_diagnostic(
        self,
        *,
        mode: str,
        question: str,
        evidence: dict[str, list[dict]],
        selected_skills: list[str],
        llm_summary: str | None,
    ) -> Diagnostic:
        """Assemble the structured Diagnostic for an ask/analyze run.

        Field mapping (human section → JSON field):

            Summary            → summary
            Primary Diagnosis  → root_cause_hypotheses[0]
            Evidence           → primary_findings[*].evidence
            Confidence         → confidence
            Recommended Action → recommendation
            Verify             → verification_command

        ``primary_findings`` are converted from the skill rows this run
        already collected, via each skill's ``to_findings_fn`` converter —
        the expensive profile queries are not re-executed.
        """
        selected_skills = list(dict.fromkeys(skill for skill in selected_skills if skill))
        diagnosis_row = self._first_actionable_row(evidence.get("root_cause_matcher", []))
        diagnosis = self._primary_diagnosis(question, evidence, diagnosis_row)
        confidence, _label = self._confidence_breakdown(evidence, diagnosis_row)
        action = self._recommended_action(diagnosis_row)
        verify_skill = self._choose_verify_skill(evidence, selected_skills)
        verify_command = self._verify_command(verify_skill) or self._VERIFY_FALLBACK_COMMAND
        summary = self._answer_summary(selected_skills, llm_summary)
        profile_id = self._profile_id()
        findings = self._evidence_findings(evidence, profile_id)
        return Diagnostic(
            id=self._diagnostic_id(profile_id, mode, question),
            summary=summary,
            recommendation=action,
            verification_command=verify_command,
            confidence=confidence,
            primary_findings=findings,
            root_cause_hypotheses=[diagnosis],
        )

    def _render_answer_text(self, diagnostic: Diagnostic) -> str:
        """Render the evidence-first Markdown answer from a Diagnostic.

        Every section, including per-row evidence citations and the confidence
        label, is derived only from fields serialized by ``Diagnostic``. A
        loaded ``diagnostics.json`` can therefore reproduce this block without
        the original skill rows.
        """
        confidence = self._confidence_text(diagnostic.confidence)
        evidence_lines = self._diagnostic_evidence_lines(diagnostic)
        diagnosis = (
            diagnostic.root_cause_hypotheses[0]
            if diagnostic.root_cause_hypotheses
            else "No diagnosis recorded."
        )

        lines = [
            "## Summary",
            diagnostic.summary,
            "",
            "## Primary Diagnosis",
            diagnosis,
            "",
            "## Evidence",
        ]
        if evidence_lines:
            lines.extend(evidence_lines)
        else:
            lines.append("- No structured EvidenceRow is embedded in primary_findings.")
        lines.extend(
            [
                "",
                "## Confidence",
                confidence,
                "",
                "## Recommended Action",
                diagnostic.recommendation,
                "",
                "## Verify",
            ]
        )
        if diagnostic.verification_command == self._VERIFY_FALLBACK_COMMAND:
            lines.append(
                "Could not build a runnable verification command from structured "
                "findings. Inspect available skills with:"
            )
        lines.append(f"`{diagnostic.verification_command}`")
        return "\n".join(lines)

    def _diagnostic_id(self, profile_id: str, mode: str, question: str) -> str:
        """Deterministic diagnostic id from profile, request, and analysis scope.

        Two runs over the same profile with the same mode, question, and trim
        scope produce the same id. Scope is included because a full-profile
        run and a trimmed run can produce different findings and verification
        commands.
        """
        scope = ",".join(
            f"{key}={value}" for key, value in sorted(self._trim_kwargs.items())
        ) or "full-profile"
        digest = hashlib.sha256(
            f"{profile_id}|{mode}|{question}|{scope}".encode()
        ).hexdigest()
        return f"diag_{digest[:16]}"

    def _profile_id(self) -> str:
        """Resolve (and cache) the content-derived profile id.

        Mirrors ``EvidenceBuilder``: reads the META_DATA / TARGET_INFO
        tables from the original SQLite connection (``profile.conn``), not
        the parquet cache, and falls back to a path-derived id when those
        tables are unreachable. Diagnostics must never crash the answer
        path, so any failure degrades to the path-derived id.
        """
        if self._cached_profile_id is None:
            from ..fingerprint import get_profile_id

            id_conn = (
                getattr(self.profile, "conn", None) if self.profile is not None else self.conn
            )
            try:
                self._cached_profile_id = get_profile_id(id_conn, fallback_path=self.profile_path)
            except Exception:
                log.debug("profile id resolution failed; using path fallback", exc_info=True)
                digest = hashlib.sha256(str(self.profile_path).encode("utf-8")).hexdigest()
                self._cached_profile_id = f"nsys1:path:{digest}"
        return self._cached_profile_id

    def _evidence_findings(
        self,
        evidence: dict[str, list[dict]],
        profile_id: str,
    ) -> list:
        """Convert already-collected skill rows into v0.1 Findings.

        Reuses each skill's ``to_findings_fn`` converter over the rows this
        run already executed — no profile query is re-run and no
        ``EvidenceBuilder`` pass is needed. Skills without a converter are
        skipped. Bounded by ``_MAX_PRIMARY_FINDINGS``; ordering follows
        evidence insertion order, so the result is deterministic.
        """
        from ..evidence_builder import _invoke_to_findings

        context = {"profile_id": profile_id}
        findings: list = []
        for skill_name, rows in evidence.items():
            if not rows:
                continue
            skill = get_skill(skill_name)
            if skill is None or skill.to_findings_fn is None:
                continue
            try:
                findings.extend(_invoke_to_findings(skill.to_findings_fn, rows, context))
            except Exception as e:
                log.debug("to_findings_fn for '%s' failed: %s", skill_name, e, exc_info=True)
        return findings[: self._MAX_PRIMARY_FINDINGS]

    def _answer_summary(
        self,
        selected_skills: list[str],
        llm_summary: str | None,
    ) -> str:
        """Return model synthesis when available, otherwise a deterministic summary."""
        if llm_summary:
            lines = [line.strip() for line in str(llm_summary).strip().splitlines()]

            summary_lines = []
            for line in lines:
                if line.startswith("#"):
                    if summary_lines:
                        break
                    continue
                if line:
                    summary_lines.append(line)

            summary = " ".join(summary_lines)
            if summary and not summary.startswith("(LLM synthesis failed:"):
                return summary

        ran = ", ".join(skill for skill in selected_skills if skill)
        if ran:
            return (
                f"Ran {ran} against the profile and summarized the strongest supported "
                "signal in a verification-friendly format."
            )
        return (
            "No skill returned usable evidence, so the answer is limited to a "
            "verification fallback."
        )

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

    def _confidence_breakdown(
        self, evidence: dict[str, list[dict]], diagnosis_row: dict | None
    ) -> tuple[float, str]:
        """Return the (numeric confidence, human label) pair for this evidence state.

        ``diagnostics.json`` carries the float (``Diagnostic.confidence``);
        the Markdown answer carries the label. Both come from this one
        function so the two can never diverge.
        """
        row_count = sum(len(rows) for rows in evidence.values())
        if diagnosis_row and row_count:
            severity = str(diagnosis_row.get("severity", "")).strip().lower()
            confidence_by_severity = {
                "critical": 0.90,
                "warning": 0.75,
                "info": 0.55,
            }
            if severity in confidence_by_severity:
                value = confidence_by_severity[severity]
                return value, self._confidence_text(value)
            return 0.65, self._confidence_text(0.65)
        if row_count:
            return 0.60, self._confidence_text(0.60)
        return 0.20, self._confidence_text(0.20)

    @staticmethod
    def _confidence_text(value: float) -> str:
        """Format a confidence value using only data persisted in Diagnostic."""
        if value >= 0.85:
            band = "high"
        elif value >= 0.70:
            band = "medium-high"
        elif value >= 0.50:
            band = "medium"
        else:
            band = "low"
        return f"{value:.2f} ({band})"

    def _diagnostic_evidence_lines(self, diagnostic: Diagnostic) -> list[str]:
        """Render every embedded EvidenceRow from ``primary_findings``."""
        lines: list[str] = []
        for finding in diagnostic.primary_findings:
            for row in finding.evidence or []:
                metric = self._evidence_values_fragment(row.values, row.units)
                window = self._finding_window_fragment(finding)
                scope = self._finding_scope_fragment(finding)
                lines.append(
                    f"- evidence_id={row.id}; source_skill={row.source_skill}; "
                    f"finding={self._compact_value(finding.label)}; metric={metric}; "
                    f"window={window}; scope={scope}"
                )
        return lines

    def _evidence_values_fragment(self, values: dict, units: dict) -> str:
        parts: list[str] = []
        for key, value in values.items():
            unit = units.get(key, "")
            parts.append(f"{key}={self._compact_value(value)}{unit}")
        return ", ".join(parts) if parts else "row_present=true"

    def _compact_value(self, value) -> str:
        text = str(value)
        return text if len(text) <= 120 else text[:117] + "..."

    @staticmethod
    def _finding_window_fragment(finding) -> str:
        selection = finding.selection
        start = selection.start_ns if selection is not None else finding.start_ns
        end = selection.end_ns if selection is not None else finding.end_ns
        if start is not None and end is not None:
            return f"{start}-{end}ns"
        return "full profile"

    @staticmethod
    def _finding_scope_fragment(finding) -> str:
        selection = finding.selection
        parts: list[str] = []
        if selection is not None:
            for key, values in (
                ("gpu_ids", selection.gpu_ids),
                ("rank_ids", selection.rank_ids),
                ("stream_ids", selection.stream_ids),
                ("nvtx_path", selection.nvtx_path),
            ):
                if values:
                    parts.append(f"{key}={','.join(str(value) for value in values)}")
        if not parts and finding.gpu_id is not None:
            parts.append(f"gpu_id={finding.gpu_id}")
        if not parts and finding.stream is not None:
            parts.append(f"stream={finding.stream}")
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

    def _try_llm_synthesis(
        self,
        question: str,
        evidence: dict[str, list[dict]],
        *,
        summary_only: bool = False,
    ) -> str | None:
        """Try to use an LLM to synthesize an answer from structured evidence.

        Args:
            question: The question to answer.
            evidence: Dict mapping skill names to their JSON-serializable results.
            summary_only: Return one concise summary paragraph for the
                deterministic ask/analyze answer formatter.

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
        response_instruction = ""
        max_tokens = 2048
        if summary_only:
            response_instruction = (
                "\n\nReturn only one concise executive-summary paragraph grounded in the "
                "provided evidence. Do not include a heading or any other answer sections; "
                "the caller will add the diagnosis, evidence, confidence, action, and verify "
                "sections deterministically."
            )
            max_tokens = 256

        user_msg = (
            f"Profile analysis data (structured JSON):\n"
            f"```json\n{evidence_json}\n```\n\n"
            f"Based on this data, answer the following question:\n{question}"
            f"{response_instruction}"
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
                    max_tokens=max_tokens,
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
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return message.content[0].text
        except Exception as e:
            log.debug("LLM synthesis (anthropic) failed: %s", e, exc_info=True)
            return f"(LLM synthesis failed: {e})"
