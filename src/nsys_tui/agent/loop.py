"""
loop.py — Core agent analysis loop.

The Agent takes a profile, selects relevant skills, executes them,
and produces a structured analysis report. Works without LLM by default
(keyword-based skill selection + template reporting). With the [agent]
extra installed, can delegate to an LLM for natural language analysis.

Supports conversation mode for follow-up questions — the Agent maintains
a message history so the LLM has full context of prior Q&A.
"""
import sqlite3
from typing import Optional

from ..skills.registry import all_skills, get_skill, run_skill


class Agent:
    """GPU profile analysis agent.

    Usage:
        agent = Agent("profile.sqlite")
        report = agent.analyze()         # auto-report
        answer = agent.ask("why slow?")  # targeted question

        # Conversation mode (follow-up questions):
        answer1 = agent.ask("what are the top kernels?")
        answer2 = agent.ask("tell me more about the slowest one")
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
        "iteration": ["top_kernels", "gpu_idle_gaps", "nvtx_kernel_map"],
        "iter": ["top_kernels", "gpu_idle_gaps", "nvtx_kernel_map"],
        "latency": ["kernel_launch_overhead", "gpu_idle_gaps"],
        "bottleneck": ["top_kernels", "gpu_idle_gaps", "thread_utilization"],
        "communication": ["nccl_breakdown"],
        "bandwidth": ["memory_transfers"],
    }

    def __init__(self, profile_path: str):
        self.path = profile_path
        self.conn = sqlite3.connect(profile_path)
        self._conversation: list[dict] = []  # LLM message history

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
        and skill results as context. Maintains conversation history so
        follow-up questions have full context.

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

        # Gather evidence from skills
        evidence_parts = []
        for skill_name in selected:
            try:
                result = run_skill(skill_name, self.conn)
                evidence_parts.append(result)
            except Exception as e:
                evidence_parts.append(f"({skill_name}: skipped — {e})")

        evidence_text = "\n\n".join(evidence_parts)

        # Try LLM synthesis if available (with conversation history)
        llm_answer = self._try_llm_synthesis(question, evidence_text)
        if llm_answer:
            return llm_answer

        # Fallback: return raw skill output
        sections = [f"Question: {question}\n"]
        sections.append(evidence_text)
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

    def _try_llm_synthesis(self, question: str, evidence: str) -> Optional[str]:
        """Try to use an LLM to synthesize an answer. Returns None if no LLM available.

        Maintains conversation history in self._conversation so follow-up
        questions have full context of prior Q&A.
        """
        try:
            import anthropic
        except ImportError:
            return None

        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        try:
            from .persona import build_system_prompt
            client = anthropic.Anthropic(api_key=api_key)

            user_content = (
                f"Here is data from an Nsight Systems profile analysis:\n\n"
                f"{evidence}\n\n"
                f"Based on this data, answer the following question:\n{question}"
            )

            # Add to conversation history
            self._conversation.append({"role": "user", "content": user_content})

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=build_system_prompt(),
                messages=self._conversation,
            )

            assistant_text = message.content[0].text

            # Store assistant response for follow-up context
            self._conversation.append({"role": "assistant", "content": assistant_text})

            return assistant_text
        except Exception as e:
            return f"(LLM synthesis failed: {e})"

    @property
    def has_conversation(self) -> bool:
        """Whether there is an active conversation with prior exchanges."""
        return len(self._conversation) > 0

    def reset_conversation(self):
        """Clear conversation history to start fresh."""
        self._conversation.clear()
