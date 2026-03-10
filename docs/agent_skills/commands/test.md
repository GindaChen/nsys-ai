# /nsys:test — Run Test Suite

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Run the nsys-ai skill test suite and update `skills/TEST.md` with results.
Analogous to `cli-anything:test` — tests must pass before results are recorded.

---

## Usage

```
/nsys:test [scope]
```

- `scope` (optional):
  - Omit → run full test suite (`pytest tests/ -v --tb=short`)
  - `"cli"` → only smoke-test the CLI entry point
  - `"mfu"` → only run MFU-related tests
  - `"chat"` → only run agent/chat tests
  - `"all"` → equivalent to omit

---

## What This Command Does

### Step 1: Locate Test Suite
Verify the test directory exists:
```bash
ls tests/
```
Expected: `test_cli.py`, `test_chat.py`, `test_region_mfu.py`, etc.

### Step 2: Run Smoke Test First
```bash
python -m nsys_ai --help
```
If this fails → stop. Do NOT run the full suite. Tell user the package is not installed.
Fix: `pip install -e '.[dev]'`

### Step 3: Run the Test Suite
```bash
pytest tests/ -v --tb=short
```
Capture the full output. Do not truncate.

### Step 4: Check Pass Rate
- `100%` passed → proceed to Step 5
- Any failures → **do NOT update `skills/TEST.md`**. Show which tests failed.
  Offer to investigate the failures before re-running.

### Step 5: Update `skills/TEST.md` (only if 100% pass)
Append to the `## Test Results` section:
```markdown
### Run: <date> <time>
**Result**: <N> passed in <X>s
**Command**: pytest tests/ -v --tb=short

<full pytest output>
```

### Step 6: Report Summary
```
Test run: <date>
Scope:    <full / mfu / cli / chat>
Result:   <N> passed, <M> failed in <X>s
TEST.md:  <updated / NOT updated (failures present)>
```

---

## Failure Handling

| Failure type | Action |
|-------------|--------|
| Module import error | Check `pip install -e '.[dev]'` |
| `test_cli.py` failure | Likely a CLI API change; check `__main__.py` |
| `test_chat.py` failure | LLM API key issue or agent loop logic change |
| `test_region_mfu.py` | MFU calculation regression; check `region_mfu.py` |
| Any failure | Do NOT update `skills/TEST.md`. Show failures and stop. |

---

## Success Criteria

- All tests pass (100% pass rate)
- `python -m nsys_ai --help` works without error
- `skills/TEST.md` appended with timestamped results
- TEST.md was only updated because all tests passed (not just "ran")
