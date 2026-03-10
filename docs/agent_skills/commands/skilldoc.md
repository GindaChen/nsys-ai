# /nsys:skilldoc — Validate Skill System Integrity

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Validate that the `docs/agent_skills/` directory is internally consistent and
complete. Analogous to `cli-anything:validate` — checks structure, not analysis output.

Use `/nsys:validate` to validate an **analysis result**.
Use `/nsys:skilldoc` to validate the **skill system itself**.

---

## Usage

```
/nsys:skilldoc [focus]
```

- `focus` (optional): limit scope
  - Omit → run all 30 checks
  - `"skills"` → only check `skills/*.md` files
  - `"commands"` → only check `commands/*.md` files
  - `"loader"` → only test `prompt_loader.py` imports

---

## What This Command Does

For each check: mark ✓ (pass) or ✗ (fail) and explain why.
**Do not skip any category.**

---

## Validation Checklist (30 checks)

### Category 1: Directory Structure (5 checks)

- [ ] S1.1 `docs/agent_skills/PRINCIPLES.md` exists
- [ ] S1.2 `docs/agent_skills/INDEX.md` exists
- [ ] S1.3 `docs/agent_skills/skills/` directory exists with ≥ 5 skill files
- [ ] S1.4 `docs/agent_skills/commands/` directory exists with ≥ 4 command files
- [ ] S1.5 `src/nsys_ai/prompt_loader.py` exists and imports without error

### Category 2: PRINCIPLES.md Content (5 checks)

- [ ] S2.1 Has `## Key Principles` section
- [ ] S2.2 Has `## Non-Negotiable Rules` section
- [ ] S2.3 Has `## Error Handling` table
- [ ] S2.4 Has `## Skill File List` routing table
- [ ] S2.5 Has `## Acceptance Criteria` checklist

### Category 3: Skill Files — Structure (5 checks per skill, sample 2)

For each of `skills/mfu.md` and `skills/triage.md`:

- [ ] S3.1 File begins with a one-sentence purpose statement
- [ ] S3.2 Has at least one numbered step-by-step workflow block
- [ ] S3.3 Has `## Acceptance Criteria` section with verifiable checkboxes
- [ ] S3.4 Has `## Error Handling` table or equivalent
- [ ] S3.5 References `PRINCIPLES.md` (does not duplicate its content)

### Category 4: INDEX.md Routing (5 checks)

- [ ] S4.1 Every file in `skills/*.md` (excluding SKILL_TEMPLATE, TEST) has a row in the Skill Router table
- [ ] S4.2 Every file in `commands/*.md` has a row in the Slash Commands table
- [ ] S4.3 All links in INDEX.md resolve (no broken relative paths)
- [ ] S4.4 Slash Commands table has ≥ 4 entries
- [ ] S4.5 Skill Router table has ≥ 5 entries

### Category 5: prompt_loader.py Integration (5 checks)

- [ ] S5.1 `load_skill("PRINCIPLES.md")` returns non-empty string
- [ ] S5.2 `load_skill("skills/mfu.md")` returns non-empty string
- [ ] S5.3 `load_skill("skills/nonexistent.md")` returns `""` (graceful degradation)
- [ ] S5.4 `load_principles()` is equivalent to `load_skill("PRINCIPLES.md")`
- [ ] S5.5 `load_skill_context(["skills/mfu.md", "skills/triage.md"])` returns concatenated content

### Category 6: TEST.md Currency (5 checks)

- [ ] S6.1 `skills/TEST.md` exists
- [ ] S6.2 Has `## Part 1: Test Plan` section
- [ ] S6.3 Has `## Part 2: Test Results` section with ≥ 1 recorded run
- [ ] S6.4 Most recent recorded run has exit code 0
- [ ] S6.5 Most recent run date is within the last 30 days

---

## Validation Report Format

```
nsys-ai Skill System Validation Report
=======================================
Scope:   full / skills / commands / loader

Category 1: Directory Structure      (X/5  passed)
Category 2: PRINCIPLES.md Content    (X/5  passed)
Category 3: Skill File Structure     (X/10 passed)
Category 4: INDEX.md Routing         (X/5  passed)
Category 5: prompt_loader.py         (X/5  passed)
Category 6: TEST.md Currency         (X/5  passed)
-----------------------------------------------
Overall: X/35 checks passed

Status: PASS ✓
  — or —
Status: FAIL ✗  (N checks failed)
Failed: S4.2 commands/skilldoc.md missing from INDEX.md, ...
```

---

## How to Run S5 Checks

```python
# From the repo root, with .venv active:
python3 -c "
import nsys_ai.prompt_loader as pl
print(bool(pl.load_skill('PRINCIPLES.md')))
print(bool(pl.load_skill('skills/mfu.md')))
print(pl.load_skill('skills/nonexistent.md') == '')
print(pl.load_skill_context(['skills/mfu.md', 'skills/triage.md'])[:80])
"
```

---

## Success Criteria

- All 35 checks pass
- No broken links in INDEX.md
- `prompt_loader.py` returns expected content for all skill files
- TEST.md has a recent passing run recorded
