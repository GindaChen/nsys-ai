"""
loop.py — Core agent analysis loop.

The Agent takes a profile, selects relevant skills, executes them,
and produces a structured analysis report. Works without LLM by default
(keyword-based skill selection + template reporting). With the [agent]
extra installed, can delegate to an LLM for natural language analysis.
"""

import sqlite3

from ..skills.registry import run_skill


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
        "stall": ["gpu_idle_gaps"],
        "memory": ["memory_transfers"],
        "transfer": ["memory_transfers"],
        "h2d": ["memory_transfers"],
        "copy": ["memory_transfers"],
        "nccl": ["nccl_breakdown"],
        "allreduce": ["nccl_breakdown"],
        "collective": ["nccl_breakdown"],
        "distributed": ["nccl_breakdown"],
        "multi-gpu": ["nccl_breakdown"],
        "nvtx": ["nvtx_kernel_map"],
        "source": ["nvtx_kernel_map"],
        "attribution": ["nvtx_kernel_map"],
        "mapping": ["nvtx_kernel_map"],
        "launch": ["kernel_launch_overhead"],
        "overhead": ["kernel_launch_overhead"],
        "cpu": ["thread_utilization"],
        "thread": ["thread_utilization"],
        "utilization": ["thread_utilization"],
        "schema": ["schema_inspect"],
        "table": ["schema_inspect"],
        "mfu": ["top_kernels"],
        "flops": ["top_kernels"],
    }

    def __init__(self, profile_path: str):
        self.path = profile_path
        self.conn = sqlite3.connect(profile_path)

    def close(self):
        self.conn.close()

    def analyze(self) -> str:
        """Run a full auto-analysis of the profile.

        Executes the core skills in order:
        1. schema_inspect — understand available data
        2. top_kernels — identify hotspots
        3. gpu_idle_gaps — find pipeline bubbles
        4. memory_transfers — check data movement
        5. nccl_breakdown — check collective overhead (if present)
        6. kernel_launch_overhead — check dispatch latency

        Returns:
            Formatted multi-section report.
        """
        sections = []
        sections.append("═══ nsys-ai Auto-Analysis Report ═══\n")

        # Always run these core skills
        core_skills = [
            "top_kernels",
            "gpu_idle_gaps",
            "memory_transfers",
            "nccl_breakdown",
            "kernel_launch_overhead",
        ]

        for skill_name in core_skills:
            try:
                result = run_skill(skill_name, self.conn)
                sections.append(result)
                sections.append("")
            except Exception as e:
                sections.append(f"({skill_name}: skipped — {e})\n")

        sections.append("═══ End of Report ═══")
        return "\n".join(sections)

    def ask(self, question: str) -> str:
        """Answer a natural language question about the profile.

        Without LLM: uses keyword matching to select relevant skills,
        runs them, and presents the raw output.

        With [agent] extra: delegates to LLM with the full system prompt
        and skill results as context.

        Args:
            question: Natural language question (e.g. "why is iteration 3 slow?")

        Returns:
            Analysis text.
        """
        # Select relevant skills based on keywords
        selected = self._select_skills(question)

        if not selected:
            # Default to overview skills
            selected = ["top_kernels", "gpu_idle_gaps"]

        sections = [f"Question: {question}\n"]

        for skill_name in selected:
            try:
                result = run_skill(skill_name, self.conn)
                sections.append(result)
                sections.append("")
            except Exception as e:
                sections.append(f"({skill_name}: skipped — {e})\n")

        # Try LLM synthesis if available
        llm_answer = self._try_llm_synthesis(question, sections)
        if llm_answer:
            sections.append("\n── AI Analysis ──")
            sections.append(llm_answer)

        return "\n".join(sections)

    def run_skill(self, name: str, **kwargs) -> str:
        """Run a specific skill by name."""
        return run_skill(name, self.conn, **kwargs)

    def _select_skills(self, question: str) -> list[str]:
        """Select skills relevant to a question using keyword matching."""
        q_lower = question.lower()
        selected = set()
        for keyword, skill_names in self._KEYWORD_MAP.items():
            if keyword in q_lower:
                selected.update(skill_names)
        return sorted(selected)

    def _try_llm_synthesis(self, question: str, evidence_sections: list[str]) -> str | None:
        """Try to use an LLM to synthesize an answer. Returns None if no LLM available."""
        import os

        evidence = "\n".join(evidence_sections)
        user_msg = (
            f"Here is data from an Nsight Systems profile analysis:\n\n"
            f"{evidence}\n\n"
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
            return f"(LLM synthesis failed: {e})"

