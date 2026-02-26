"""
annotator.py — Insert NVTX annotations into Python source files.

Works at the text/AST level to wrap function calls and code blocks with
env-gated NVTX ranges. Designed to be used by the Nsight AI agent loop
to iteratively refine kernel-to-source mapping.

The annotator:
1. Reads a Python source file
2. Identifies function definitions and call sites
3. Wraps them with nsight_range() context managers
4. Writes the modified source back (or to a new file)

All inserted annotations use the gate.py mechanism — they're no-ops
unless NSIGHT_AI=1 is set in the environment.
"""
import ast
import textwrap
from typing import Optional


# ── The import line we inject at the top of files ──────────────────

IMPORT_LINE = "from nsys_tui.ai.gate import nsight_range  # auto-inserted by nsight-ai"


# ── Simple text-based annotation (no AST rewrite) ─────────────────

def annotate_function_calls(source: str, target_func: str,
                            context: str = "") -> str:
    """
    Wrap calls to `target_func(...)` with nsight_range().

    This is a simple text-based approach — it finds lines containing
    `target_func(` and wraps them. For complex cases (multi-line calls,
    nested expressions), use the AST-based approach.

    Args:
        source: Python source code string
        target_func: Function name to wrap (e.g. "self.attention")
        context: Optional prefix for the NVTX name (e.g. "TransformerLayer")

    Returns:
        Modified source code with NVTX annotations inserted.
    """
    lines = source.split("\n")
    result = []
    needs_import = IMPORT_LINE not in source

    if needs_import:
        # Insert import after the last existing import
        import_inserted = False
        for i, line in enumerate(lines):
            result.append(line)
            if not import_inserted and (line.startswith("import ") or
                                         line.startswith("from ")):
                # Check if next line is NOT an import
                if i + 1 >= len(lines) or not (lines[i+1].startswith("import ") or
                                                 lines[i+1].startswith("from ") or
                                                 lines[i+1].strip() == ""):
                    result.append(IMPORT_LINE)
                    import_inserted = True
        if not import_inserted:
            result.insert(0, IMPORT_LINE)
        lines = result
        result = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        if f"{target_func}(" in stripped and "nsight_range" not in line:
            label = f"{context}.{target_func}" if context else target_func
            result.append(f"{indent}with nsight_range(\"{label}\"):")
            result.append(f"{indent}    {stripped}")
        else:
            result.append(line)

    return "\n".join(result)


# ── AST-based function body annotation ─────────────────────────────

def annotate_function_body(source: str, func_name: str,
                           class_name: Optional[str] = None) -> str:
    """
    Wrap the entire body of a function with nsight_range().

    Finds `def func_name(...)` (optionally inside `class class_name`)
    and wraps its body.

    Args:
        source: Python source code
        func_name: Name of the function to annotate
        class_name: If set, only annotate this function inside this class

    Returns:
        Modified source with the function body wrapped.
    """
    tree = ast.parse(source)
    lines = source.split("\n")

    # Find the function
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if class_name:
                # Check parent is the right class (heuristic: check enclosing)
                # AST doesn't track parent, so we search within class bodies
                pass  # TODO: implement class scoping
            target = node
            break

    if not target or not target.body:
        return source

    # Get the indentation of the first line of the function body
    first_body_line = target.body[0].lineno - 1  # 0-indexed
    body_indent = lines[first_body_line][:len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]

    label = f"{class_name}.{func_name}" if class_name else func_name

    # Check if already annotated
    if "nsight_range" in lines[first_body_line]:
        return source

    # Insert the with statement
    with_line = f'{body_indent}with nsight_range("{label}"):'
    # Indent all body lines by one level
    last_body_line = target.body[-1].end_lineno - 1  # 0-indexed, inclusive

    new_lines = lines[:first_body_line]
    new_lines.append(with_line)
    for i in range(first_body_line, last_body_line + 1):
        new_lines.append("    " + lines[i])
    new_lines.extend(lines[last_body_line + 1:])

    # Ensure import exists
    result = "\n".join(new_lines)
    if IMPORT_LINE not in result:
        result = IMPORT_LINE + "\n" + result

    return result


# ── Bulk annotation ────────────────────────────────────────────────

def annotate_all_methods(source: str, class_name: str) -> str:
    """
    Wrap every method body in a class with nsight_range().

    Each method gets labeled as "ClassName.method_name".
    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = [n.name for n in node.body
                       if isinstance(n, ast.FunctionDef)
                       and not n.name.startswith("_")]

    # Apply annotations one at a time (each modifies source)
    for method in reversed(methods):  # reversed to preserve line numbers
        source = annotate_function_body(source, method, class_name)

    return source
