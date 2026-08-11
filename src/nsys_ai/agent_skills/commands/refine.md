# /nsys:refine — Add or Improve a Skill

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Use this command to add a new analysis skill or improve an existing one.
Analogous to `cli-anything:refine` in CLI-Anything.

---

## Usage

```
/nsys:refine [skill-name] [description]
```

- `skill-name`: existing skill to refine, or name for a new skill
- `description`: what capability to add or improve

---

## What This Command Does

### Step 1: Inventory Current Coverage
- Read `PRINCIPLES.md` Skill File List table
- Read the target `skills/<name>.md` if it exists
- Identify what workflows exist vs. what the user wants

### Step 2: Gap Analysis
Does the request map to an **existing skill**?
- Yes → find the step to add or improve; modify `skills/<name>.md`
- No → create `skills/<name>.md` using `skills/SKILL_TEMPLATE.md`

What **tools** does the new capability require?
- Set A tools (single profile): always available
- Set B tools (diff mode): only available when two profiles are loaded

### Step 3: Implement
- Follow `skills/SKILL_TEMPLATE.md` for new skills
- Add SQL queries with exact column names (no `SELECT *`)
- Include acceptance criteria specific to this workflow
- State which tool(s) are called and in what order

### Step 4: Register
- Add to Skill File List in `PRINCIPLES.md`
- Add routing entry in `INDEX.md`
- If warranted, create `commands/<name>.md`

### Step 5: Verify
Run the Acceptance Checklist from `PRINCIPLES.md`.
If adding a tool call, verify:
- The tool name exists in Set A or Set B (see `PRINCIPLES.md`)
- The output fields used are real (check from a test run or the tool docstring)

---

## Success Criteria

- New skill follows the `SKILL_TEMPLATE.md` format
- Acceptance Criteria section added to the new skill
- Entry added to `PRINCIPLES.md` Skill File List
- Entry added to `INDEX.md` routing table
