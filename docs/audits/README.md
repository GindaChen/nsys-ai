# Skill audits

Notes from running built-in skills on public Nsight SQLite profiles.

## L40S FastVideo (`perf_compile.sqlite`)

- [l40s-fastvideo-skill-audit.md](l40s-fastvideo-skill-audit.md) — 35 skills, grades, gaps
- [l40s-fastvideo-gaps.md](l40s-fastvideo-gaps.md) — follow-up issues

Dataset: [rich7421/fastvideo-wan-l40s-nsys](https://huggingface.co/datasets/rich7421/fastvideo-wan-l40s-nsys). Use `profiles/perf_compile.sqlite` for the audit.

```bash
hf download rich7421/fastvideo-wan-l40s-nsys --repo-type dataset \
  --local-dir ~/.cache/nsys-ai-datasets/fastvideo-wan-l40s-nsys

export L40S_PROFILE=~/.cache/nsys-ai-datasets/fastvideo-wan-l40s-nsys/profiles/perf_compile.sqlite

pip install -e '.[dev]'
python scripts/batch_audit_skills.py "$L40S_PROFILE"
```

Default output: `audit/l40s-perf_compile/`. Pass a second argument to use another directory under `audit/`. Everything under `audit/` is gitignored except `audit/.gitignore` — do not commit batch JSON or the profile.

Reference numbers live in the dataset’s `analysis_measurements.md` and in the audit table.
