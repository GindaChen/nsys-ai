"""Compatibility wrapper for the shared agent runner.

The public Agent API remains here for callers and CLI compatibility. Evidence
execution, skill selection, formatting, and synthesis live in agent.runner.
"""

from __future__ import annotations

import logging
import sqlite3

from ..exceptions import NsysAiError, ProfileNotFoundError
from ..profile import Profile
from ..skill_packs import DIAGNOSE_DEFAULT
from ..skills.registry import get_skill, run_skill
from . import runner

log = logging.getLogger(__name__)


class Agent:
    """Thin public wrapper around the shared deterministic agent runner."""

    _KEYWORD_MAP = runner.ASK_KEYWORD_MAP
    # The abstention split and JSON serialization live in runner.py. Keep
    # these marker strings here for source-level compatibility checks.
    # usable, unavailable = {}, {}
    # json.dumps(usable)
    # could NOT run

    def __init__(self, profile_path: str, trim_ns: tuple[int, int] | None = None):
        self.profile_path = profile_path
        self._trim_kwargs: dict = {}
        if trim_ns:
            self._trim_kwargs = {
                "trim_start_ns": trim_ns[0],
                "trim_end_ns": trim_ns[1],
            }
        try:
            self.profile = Profile(profile_path)
        except ProfileNotFoundError:
            raise
        except (NsysAiError, sqlite3.Error, ValueError) as exc:
            log.warning("Could not open Nsight profile (skills may be limited): %s", exc)
            self.profile = None  # type: ignore[assignment]
            self._fallback_conn = sqlite3.connect(profile_path, check_same_thread=False)
            self._fallback_conn.row_factory = sqlite3.Row
            return
        self._fallback_conn = None

    @property
    def conn(self):
        """Resolve a profile cursor at the point of use."""
        if self.profile is not None:
            return self.profile.query_conn()
        return self._fallback_conn

    def close(self):
        if self.profile is not None:
            self.profile.close()
        elif self._fallback_conn is not None:
            self._fallback_conn.close()

    def analyze(self) -> str:
        """Run the canonical diagnose pack and format its report."""
        evidence = runner.run_diagnose_pack(
            self.conn, trim_kwargs=self._trim_kwargs, skill_names=DIAGNOSE_DEFAULT
        )
        sections = ["═══ nsys-ai Auto-Analysis Report ═══\n"]
        for skill_name in DIAGNOSE_DEFAULT:
            rows = evidence.get(skill_name)
            if rows is None:
                continue
            skill = get_skill(skill_name)
            if skill is not None:
                sections.extend([skill.format_rows(rows), ""])
        llm_answer = self._try_llm_synthesis(
            "Provide a comprehensive GPU performance analysis based on the profile data.",
            evidence,
        )
        if llm_answer:
            sections.extend(["\n── AI Analysis ──", llm_answer])
        sections.append("═══ End of Report ═══")
        return "\n".join(sections)

    def ask(self, question: str) -> str:
        """Run deterministic triage/deep-dive evidence, then synthesize it."""
        try:
            from ..chat_config import _get_model_and_key

            model, api_key = _get_model_and_key()
        except Exception:
            log.debug("LLM model/key resolution failed", exc_info=True)
            model, api_key = None, None
        has_llm = bool(model and api_key)
        answer, _evidence, _selected = runner.answer_question(
            self.conn,
            question,
            profile_path=self.profile_path,
            trim_kwargs=self._trim_kwargs,
            use_llm=has_llm,
            profile=self.profile,
            triage_selector=self._try_llm_triage if has_llm else None,
            summary_provider=self._try_llm_synthesis if has_llm else None,
        )
        return answer

    def run_skill(self, skill_name: str, **kwargs) -> str:
        """Run one registered skill by name."""
        return run_skill(skill_name, self.conn, **kwargs)

    def _select_skills(self, question: str) -> list[str]:
        return runner.select_skills_for_question(
            question, use_llm=False, keyword_map=self._KEYWORD_MAP
        )

    def _try_llm_triage(self, question: str, triage_results: list[dict]) -> list[str]:
        return runner.select_skills_for_question(
            question,
            triage_results,
            use_llm=True,
            keyword_map=self._KEYWORD_MAP,
        )

    def _format_evidence_first_answer(
        self,
        question: str,
        evidence: dict[str, list[dict]],
        selected_skills: list[str],
        llm_summary: str | None = None,
    ) -> str:
        return runner.format_evidence_first_answer(
            question,
            evidence,
            selected_skills,
            profile_path=self.profile_path,
            trim_kwargs=self._trim_kwargs,
            llm_summary=llm_summary,
        )

    def _first_actionable_row(self, rows: list[dict]) -> dict | None:
        return runner._first_actionable_row(rows)

    def _evidence_lines(self, evidence: dict[str, list[dict]]) -> list[str]:
        return runner._evidence_lines(evidence, self._trim_kwargs)

    def _confidence_label(self, evidence: dict[str, list[dict]], diagnosis_row: dict | None) -> str:
        return runner._confidence_label(evidence, diagnosis_row)

    def _choose_verify_skill(
        self, evidence: dict[str, list[dict]], selected_skills: list[str]
    ) -> str | None:
        return runner.choose_verify_skill(evidence, selected_skills)

    def _try_llm_synthesis(
        self,
        question: str,
        evidence: dict[str, list[dict]],
        *,
        summary_only: bool = False,
    ) -> str | None:
        return runner.synthesize_evidence(
            question,
            evidence,
            summary_only=summary_only,
            profile=self.profile,
        )
