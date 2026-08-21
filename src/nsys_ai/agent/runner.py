"""Shared deterministic agent runner.

The runner owns profile evidence preparation, deterministic skill selection,
and evidence-first answer shaping. Transport adapters and the public Agent
wrapper provide lifecycle and presentation compatibility around it. LLM calls
in this module only select skills or summarize already fetched rows; they never
author profile SQL.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from collections.abc import Mapping

from ..skill_packs import ASK_FALLBACK, ASK_KEYWORD_MAP, DIAGNOSE_DEFAULT
from ..skills.base import is_abstention_row
from ..skills.registry import get_skill, list_skills

log = logging.getLogger(__name__)


def _is_usable_evidence_row(row: object) -> bool:
    return (
        isinstance(row, dict)
        and bool(row)
        and not is_abstention_row(row)
        and not bool(row.get("error"))
    )


def _unavailable_evidence_reason(row: Mapping) -> str | None:
    if is_abstention_row(row):
        return str(row.get("reason") or "could not run")
    if row.get("error"):
        return str(row["error"])
    return None


def _execute_pack(conn, skill_names: list[str], trim_kwargs: Mapping | None) -> dict[str, list[dict]]:
    evidence: dict[str, list[dict]] = {}
    kwargs = dict(trim_kwargs or {})
    for skill_name in skill_names:
        try:
            skill = get_skill(skill_name)
            if skill is None:
                continue
            evidence[skill_name] = skill.execute(conn, **kwargs)
        except Exception as exc:
            log.debug("Skill %s failed: %s", skill_name, exc, exc_info=True)
            evidence[skill_name] = [{"error": str(exc)}]
    return evidence


def run_diagnose_pack(
    conn,
    *,
    trim_kwargs: Mapping | None = None,
    skill_names: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run the canonical diagnose pack and return structured evidence rows."""
    return _execute_pack(conn, list(skill_names or DIAGNOSE_DEFAULT), trim_kwargs)


def _keyword_select(question: str, keyword_map: Mapping[str, list[str]]) -> list[str]:
    selected: set[str] = set()
    question_lower = question.lower()
    for keyword, skill_names in keyword_map.items():
        if keyword in question_lower:
            selected.update(skill_names)
    return sorted(selected)[:4]


def _parse_llm_skill_list(text: str, available: set[str]) -> list[str]:
    selected: list[str] = []
    for item in text.split(","):
        skill_name = item.strip().replace(chr(96), "").replace("'", "").replace('"', "")
        if skill_name in available and skill_name not in selected:
            selected.append(skill_name)
    return selected[:4]


def select_skills_for_question(
    question: str,
    triage_rows: list[dict] | None = None,
    *,
    use_llm: bool = False,
    keyword_map: Mapping[str, list[str]] = ASK_KEYWORD_MAP,
    fallback: list[str] = ASK_FALLBACK,
) -> list[str]:
    """Select at most four registered skills, falling back deterministically."""
    selected: list[str] = []
    if use_llm and triage_rows:
        triage_json = json.dumps(triage_rows, indent=2, default=str)
        available = set(list_skills())
        prompt = "".join(
            (
                "The user asked: ",
                repr(question),
                ". We ran root_cause_matcher and got:\n",
                triage_json,
                "\nSelect up to 4 names from this registry: ",
                ", ".join(sorted(available)),
                ". Reply with comma-separated names only.",
            )
        )
        try:
            import litellm

            from ..chat_config import _get_model_and_key

            model, _ = _get_model_and_key()
            if model:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                )
                selected = _parse_llm_skill_list(
                    response.choices[0].message.content.strip(), available
                )
        except Exception:
            log.debug("LLM triage failed, falling back to keywords", exc_info=True)
    if not selected:
        selected = _keyword_select(question, keyword_map)
    return selected[:4] or list(fallback)[:4]


def run_question_evidence(
    conn,
    question: str,
    *,
    trim_kwargs: Mapping | None = None,
    use_llm: bool = False,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Run triage plus selected skills; return evidence and the selected list."""
    triage = _execute_pack(conn, ["root_cause_matcher"], trim_kwargs)
    selected = select_skills_for_question(
        question,
        triage.get("root_cause_matcher", []),
        use_llm=use_llm,
    )
    selected = [skill for skill in selected if skill != "root_cause_matcher"]
    if not selected:
        selected = list(ASK_FALLBACK)[:4]
    evidence = dict(triage)
    for skill_name, rows in _execute_pack(conn, selected, trim_kwargs).items():
        evidence[skill_name] = rows
    return evidence, selected


def answer_question(
    conn,
    question: str,
    *,
    profile_path: str = "",
    trim_kwargs: Mapping | None = None,
    use_llm: bool = False,
    profile=None,
    triage_selector=None,
    summary_provider=None,
) -> tuple[str, dict[str, list[dict]], list[str]]:
    """Run the canonical ask workflow and build its evidence-first answer.

    This is the transport-neutral ask contract.  Callers provide a profile
    connection and only choose presentation details; triage, the four-skill
    cap, evidence collection, and answer shaping stay in this runner.
    ``use_llm`` permits optional planner/synthesizer calls, but the returned
    answer remains usable and deterministic when no provider is configured.
    """
    if triage_selector is None:
        evidence, selected = run_question_evidence(
            conn,
            question,
            trim_kwargs=trim_kwargs,
            use_llm=use_llm,
        )
    else:
        evidence = _execute_pack(conn, ["root_cause_matcher"], trim_kwargs)
        selected = triage_selector(question, evidence.get("root_cause_matcher", []))
        selected = [
            skill for skill in selected
            if skill and skill != "root_cause_matcher" and get_skill(skill) is not None
        ][:4]
        if not selected:
            selected = list(ASK_FALLBACK)[:4]
        evidence.update(_execute_pack(conn, selected, trim_kwargs))
    selected_with_triage = ["root_cause_matcher", *selected]
    llm_summary = None
    if use_llm:
        if summary_provider is not None:
            llm_summary = summary_provider(question, evidence, summary_only=True)
        else:
            llm_summary = synthesize_evidence(
                question, evidence, summary_only=True, profile=profile
            )
    answer = format_evidence_first_answer(
        question,
        evidence,
        selected_with_triage,
        profile_path=profile_path,
        trim_kwargs=trim_kwargs,
        llm_summary=llm_summary,
    )
    return answer, evidence, selected_with_triage


def _first_actionable_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if _is_usable_evidence_row(row):
            pattern = str(row.get("pattern", ""))
            if pattern and pattern != "No Known Anti-Patterns Detected":
                return row
    return None


def _compact_value(value: object) -> str:
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _metric_fragment(row: Mapping) -> str:
    priority = (
        "pattern", "label", "name", "kernel_name", "severity", "total_ms",
        "duration_ms", "gap_ms", "gap_ns", "idle_pct", "total_idle_ms",
        "overlap_pct", "nccl_only_ms", "compute_only_ms", "count",
    )
    parts = [
        f"{key}={_compact_value(row[key])}"
        for key in priority
        if key in row and row[key] not in (None, "")
    ]
    return ", ".join(parts[:3]) or "row_present=true"


def _window_fragment(row: Mapping, trim_kwargs: Mapping) -> str:
    start = row.get("start_ns", row.get("gpu_start_ns"))
    end = row.get("end_ns", row.get("gpu_end_ns"))
    if start is not None and end is not None:
        return f"{start}-{end}ns"
    if row.get("start_ms") is not None and row.get("end_ms") is not None:
        return f"{row['start_ms']}-{row['end_ms']}ms"
    if "trim_start_ns" in trim_kwargs and "trim_end_ns" in trim_kwargs:
        return f"{trim_kwargs['trim_start_ns']}-{trim_kwargs['trim_end_ns']}ns"
    return "full profile"


def _scope_fragment(row: Mapping) -> str:
    parts = [
        f"{key}={row[key]}"
        for key in ("gpu_id", "device_id", "device", "rank", "stream_id", "communicator_hex")
        if key in row and row[key] not in (None, "")
    ]
    return ", ".join(parts) or "profile"


def _evidence_lines(evidence: Mapping[str, list[dict]], trim_kwargs: Mapping) -> list[str]:
    lines: list[str] = []
    for skill_name, rows in evidence.items():
        for row in rows[:2]:
            if not isinstance(row, dict):
                continue
            if row.get("_summary") and len(rows) > 1:
                continue
            unavailable = _unavailable_evidence_reason(row)
            if unavailable is not None:
                lines.append(
                    f"- source_skill={skill_name}; unavailable: {unavailable.strip()}"
                )
            else:
                metric = _metric_fragment(row)
                window = _window_fragment(row, trim_kwargs)
                scope = _scope_fragment(row)
                detail = str(row.get("evidence") or row.get("note") or "").strip()
                suffix = f"; evidence={detail}" if detail else ""
                lines.append(
                    f"- source_skill={skill_name}; metric={metric}; "
                    f"window={window}; scope={scope}{suffix}"
                )
            if len(lines) >= 5:
                return lines
    return lines


def _confidence_label(evidence: Mapping[str, list[dict]], diagnosis_row: dict | None) -> str:
    row_count = sum(
        sum(_is_usable_evidence_row(row) for row in rows) for rows in evidence.values()
    )
    if diagnosis_row and row_count:
        return {
            "critical": "0.90 (high): a critical root-cause matcher finding is backed by skill output.",
            "warning": (
                "0.75 (medium-high): a warning root-cause matcher finding is backed by "
                "skill output."
            ),
            "info": (
                "0.55 (medium): an informational root-cause matcher finding is backed by "
                "skill output."
            ),
        }.get(
            str(diagnosis_row.get("severity", "")).strip().lower(),
            "0.65 (medium): a root-cause matcher finding is backed by skill output, "
            "but its severity is unknown.",
        )
    if row_count:
        return "0.60 (medium): skill output exists, but no root-cause matcher finding dominated."
    return "0.20 (low): no skill returned usable evidence."


def choose_verify_skill(
    evidence: Mapping[str, list[dict]], selected_skills: list[str]
) -> str | None:
    """Choose the first selected skill that returned usable evidence."""
    for skill_name in selected_skills:
        if any(_is_usable_evidence_row(row) for row in evidence.get(skill_name, [])):
            return skill_name
    return next(
        (
            skill_name
            for skill_name, rows in evidence.items()
            if any(_is_usable_evidence_row(row) for row in rows)
        ),
        None,
    )


def _answer_summary(
    selected_skills: list[str],
    llm_summary: str | None,
    *,
    has_usable_evidence: bool,
) -> str:
    if not has_usable_evidence:
        return (
            "I cannot answer this profile question because no selected skill returned "
            "usable evidence. See the unavailable reasons below."
        )
    if llm_summary:
        lines = [line.strip() for line in str(llm_summary).strip().splitlines()]
        if lines and lines[0].lower().lstrip("#").strip() == "summary":
            lines = lines[1:]
        summary_lines: list[str] = []
        for line in lines:
            if line.startswith("#"):
                break
            if line:
                summary_lines.append(line)
        summary = " ".join(summary_lines)
        if summary and not summary.startswith("(LLM synthesis failed:"):
            return summary
    ran = ", ".join(skill for skill in selected_skills if skill)
    return (
        f"Ran {ran} against the profile and summarized the strongest supported signal "
        "in a verification-friendly format."
        if ran
        else "No skill returned usable evidence, so the answer is limited to a verification fallback."
    )


def format_evidence_first_answer(
    question: str,
    evidence: Mapping[str, list[dict]],
    selected_skills: list[str],
    *,
    profile_path: str = "",
    trim_kwargs: Mapping | None = None,
    llm_summary: str | None = None,
) -> str:
    """Build the deterministic evidence-first answer envelope."""
    trim = dict(trim_kwargs or {})
    selected = list(dict.fromkeys(skill for skill in selected_skills if skill))
    diagnosis_row = _first_actionable_row(evidence.get("root_cause_matcher", []))
    if diagnosis_row:
        diagnosis = diagnosis_row.get("pattern") or diagnosis_row.get("label")
    else:
        diagnosis = None
        for skill_name, rows in evidence.items():
            if rows and _is_usable_evidence_row(rows[0]):
                row = rows[0]
                label = row.get("label") or row.get("name") or row.get("kernel_name")
                if label:
                    diagnosis = f"{label} ({skill_name})"
                    break
    diagnosis = diagnosis or f"No specific diagnosis could be grounded for: {question}"
    action = (
        str(diagnosis_row.get("recommendation") or diagnosis_row.get("action"))
        if diagnosis_row and (diagnosis_row.get("recommendation") or diagnosis_row.get("action"))
        else "Re-run the verify command, inspect the cited metrics and window, then collect a narrower profile with NVTX ranges if the evidence is too broad."
    )
    verify_skill = choose_verify_skill(evidence, selected)
    verify = None
    if verify_skill:
        verify = ["nsys-ai", "skill", "run", verify_skill, profile_path, "--format", "json"]
        if "trim_start_ns" in trim and "trim_end_ns" in trim:
            verify.extend(["--trim", f"{trim['trim_start_ns'] / 1e9:g}", f"{trim['trim_end_ns'] / 1e9:g}"])
        verify = " ".join(shlex.quote(str(part)) for part in verify)
    usable = any(
        any(_is_usable_evidence_row(row) for row in rows) for rows in evidence.values()
    )
    lines = [
        "## Summary", _answer_summary(selected, llm_summary, has_usable_evidence=usable),
        "", "## Primary Diagnosis", str(diagnosis), "", "## Evidence",
    ]
    lines.extend(_evidence_lines(evidence, trim) or [
        "- source_skill=none; metric=none; window=full profile; scope=profile; "
        "evidence=no skill returned usable rows"
    ])
    lines.extend(["", "## Confidence", _confidence_label(evidence, diagnosis_row),
                  "", "## Recommended Action", action, "", "## Verify"])
    mark = chr(96)
    lines.append(f"{mark}{verify}{mark}" if verify else
                 "Could not build a runnable verification command because no skill produced "
                 "evidence. Inspect available skills with:\n" + f"{mark}nsys-ai skill list{mark}")
    return "\n".join(lines)


def synthesize_evidence(
    question: str,
    evidence: Mapping[str, list[dict]],
    *,
    summary_only: bool = False,
    profile=None,
) -> str | None:
    """Ask an LLM to summarize usable evidence, never to invent evidence."""
    usable: dict[str, list[dict]] = {}
    unavailable: dict[str, str] = {}
    for skill_name, rows in evidence.items():
        real_rows = [row for row in rows if _is_usable_evidence_row(row)]
        if real_rows:
            usable[skill_name] = real_rows
        elif rows and _unavailable_evidence_reason(rows[0]) is not None:
            unavailable[skill_name] = _unavailable_evidence_reason(rows[0]) or "could not run"
        else:
            unavailable[skill_name] = "returned no rows"
    if not usable:
        return None

    evidence_json = json.dumps(usable, indent=2, default=str)
    unavailable_note = ""
    if unavailable:
        listed = "\n".join(f"- {name}: {reason}" for name, reason in unavailable.items())
        unavailable_note = (
            "\n\nThese skills could NOT run on this profile. They are not measurements "
            "and say nothing about the workload — do not draw conclusions from them.\n"
            f"{listed}"
        )
    response_instruction = ""
    max_tokens = 2048
    if summary_only:
        response_instruction = (
            "\n\nReturn only one concise executive-summary paragraph grounded in the "
            "provided evidence. Do not include a heading or other answer sections."
        )
        max_tokens = 256
    mark = chr(96)
    user_msg = (
        f"Profile analysis data (structured JSON):\n{mark * 3}json\n"
        f"{evidence_json}\n{mark * 3}\n{unavailable_note}\n\n"
        f"Based on this data, answer the following question:\n{question}{response_instruction}"
    )

    system = "You are an expert GPU profiling assistant."
    try:
        from .persona import build_system_prompt

        system = build_system_prompt()
        if profile is not None and getattr(profile, "fingerprint", None):
            system += (
                "\n\n--- TRACE CONTEXT ---\n"
                f"{profile.fingerprint.to_prompt_string()}\n---------------------"
            )
    except Exception:
        log.debug("Failed to load persona prompt", exc_info=True)

    try:
        import litellm

        model = None
        if os.environ.get("GEMINI_API_KEY"):
            model = "gemini/gemini-2.5-flash"
        elif os.environ.get("OPENAI_API_KEY"):
            model = "gpt-4o-mini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            model = "claude-sonnet-4-20250514"
        if model:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
    except ImportError:
        pass
    except Exception as exc:
        log.debug("LLM synthesis (litellm) failed: %s", exc, exc_info=True)
        return f"(LLM synthesis failed: {exc})"

    try:
        import anthropic
    except ImportError:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as exc:
        log.debug("LLM synthesis (anthropic) failed: %s", exc, exc_info=True)
        return f"(LLM synthesis failed: {exc})"
