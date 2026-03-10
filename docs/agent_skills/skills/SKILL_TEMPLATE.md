# Skill Template

Copy this file to `skills/<name>.md` when creating a new skill.
Replace all `<...>` placeholders with actual content.
Read `PRINCIPLES.md` before writing a skill; do not duplicate its content here.

---

# Skill: <Name>

Read this when the user asks: "<specific question pattern that triggers this skill>".

---

## Workflow

```
Step 1  <first tool call or SQL query>
        → <what it returns>

Step 2  <second step — with exact SQL if needed>
        Note: <any important caveat for this specific skill>

Step 3  <decision point or branching step>
        [If <condition>] → <action A>
        [If <condition>] → <action B>

Step 4  <synthesis / output step>
        Report: <what to tell the user>
```

---

## Key Tables / Reference Data

<!-- Include any lookup tables, classification tables, or reference ranges
     that this skill specifically needs. Do NOT duplicate content from PRINCIPLES.md. -->

| Column A | Column B |
|----------|---------|
| example  | value   |

---

## Acceptance Criteria

Before delivering the result, verify:

- [ ] <specific check for this skill, e.g. "MFU between 0–100%">
- [ ] <specific check, e.g. "NVTX name came from a query">
- [ ] <specific check, e.g. "Root cause states evidence field + value">

Also run the global checklist in `PRINCIPLES.md`.

---

## Error Handling

| Error | Action |
|-------|--------|
| <error condition> | <what to do> |
| <error condition> | <what to do> |
