# Command: `nsys-ai root-cause`

Manage, view, and submit root cause patterns in the local knowledge base.

`nsys-ai` maintains a store of known GPU performance anti-patterns. These patterns provide the underlying knowledge for diagnostic skills like `root_cause_matcher`.

The knowledge base is divided into three tiers:
1. **Builtin**: Shipped with the package (`docs/root-causes/book.md`).
2. **Community**: Contributed patterns (`docs/root-causes/community/*.md`).
3. **User-local**: Custom patterns stored on the current machine (`~/.nsys-ai/root-causes/`).

---

## Subcommands

### `nsys-ai root-cause list`

List all known root cause patterns across all tiers (builtin, community, and user-local).

```bash
nsys-ai root-cause list
```

**Output**: Tabular list showing Name, Severity, Source, and Tags.

### `nsys-ai root-cause show`

Display the full details of a specific root cause pattern.

```bash
nsys-ai root-cause show <name>
```

- `<name>`: A substring match for the root cause name.
- **Output**: Prints the full symptom description, underlying mechanism, detection strategy, and recommended fixes.

**Example**:
```bash
nsys-ai root-cause show "GPU Bubbles"
```

### `nsys-ai root-cause submit`

Submit a new custom root cause pattern to the user-local knowledge base.

```bash
nsys-ai root-cause submit <file.md>
```

- `<file.md>`: Path to a Markdown file containing the new pattern.

The submitted Markdown file MUST contain:
1. Valid YAML frontmatter with `name` and `severity`.
2. A Markdown section named `## Symptom`.
3. A Markdown section named `## How to Fix`.

If validation passes, the file is copied to `~/.nsys-ai/root-causes/` (or the directory specified by `--root-causes-dir`).

**Options**:
- `--root-causes-dir <dir>`: Override the default `~/.nsys-ai/root-causes/` directory for user-local files. Pass it *after* the `root-cause` subcommand (e.g., `nsys-ai root-cause --root-causes-dir /tmp list`).

---

## Relationship with `root_cause_matcher`

Note the difference between this CLI command and the `root_cause_matcher` skill:
- `nsys-ai skill run root_cause_matcher <profile>` is the **detection engine**. It runs SQL queries against a profile to detect if your GPU exhibits specific pipeline flaws.
- `nsys-ai root-cause` is the **knowledge base manager**. It defines *what* those pipeline flaws are and *how* a user should fix them once detected.
