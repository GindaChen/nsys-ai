#!/usr/bin/env python3
"""Validate or run a reproducible nsys-ai checkpoint manifest.

Examples:
    python3 scripts/checkpoint.py validate examples/checkpoints/b0-fixture/manifest.json
    python3 scripts/checkpoint.py plan examples/checkpoints/b0-fixture/manifest.json
    python3 scripts/checkpoint.py run examples/checkpoints/b0-fixture/manifest.json \
        --output /tmp/nsys-ai-checkpoint

The runner executes argv arrays from the manifest without a shell. A manifest
is a reviewed recipe, not an arbitrary command channel; inspect it before
running a profile on a machine with credentials or a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nsys_ai.checkpoint import (
    CheckpointManifestError,
    expand_command,
    load_manifest,
    resolve_profile_path,
    run_steps,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    validate = sub.add_parser("validate", help="validate manifest metadata")
    _add_manifest_options(validate)
    validate.add_argument(
        "--require-profile",
        action="store_true",
        help="also verify the referenced profile exists and matches its SHA-256",
    )
    validate.add_argument("--profile", help="override capture.profile_path for checksum verification")

    plan = sub.add_parser("plan", help="print the expanded analysis commands")
    _add_manifest_options(plan)
    plan.add_argument("--profile", help="override capture.profile_path")
    plan.add_argument("--session", help="session directory used by command templates")

    run = sub.add_parser("run", help="run the manifest analysis steps")
    _add_manifest_options(run)
    run.add_argument("--profile", help="override capture.profile_path")
    run.add_argument("--session", help="session directory used by command templates")
    run.add_argument("--output", required=True, help="directory for stdout/stderr step logs")
    run.add_argument("--timeout", type=float, default=300.0, help="per-step timeout in seconds")
    return parser


def _add_manifest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, help="path to a checkpoint JSON manifest")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used for relative paths (default: current directory)",
    )


def _load(args: argparse.Namespace, *, require_profile: bool = False) -> dict:
    return load_manifest(
        args.manifest,
        profile_root=args.repo_root,
        profile_override=getattr(args, "profile", None),
        require_profile=require_profile,
    )


def _profile(args: argparse.Namespace, manifest: dict) -> Path:
    return resolve_profile_path(
        manifest,
        repo_root=args.repo_root,
        profile_override=getattr(args, "profile", None),
    )


def _session(args: argparse.Namespace, manifest: dict) -> Path:
    if getattr(args, "session", None):
        return Path(args.session).expanduser().resolve()
    return (args.repo_root / manifest["analysis"]["session_dir"]).resolve()


def _print_error(exc: CheckpointManifestError) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "validate":
            manifest = _load(args, require_profile=args.require_profile)
            print(
                f"valid checkpoint manifest: {args.manifest} "
                f"({manifest['checkpoint']}, {manifest['schema_version']})"
            )
            return 0

        manifest = _load(args)
        profile = _profile(args, manifest)
        session = _session(args, manifest)
        if args.action == "plan":
            for step in manifest["analysis"]["steps"]:
                command = expand_command(
                    step["command"], profile=profile, repo=args.repo_root, session=session
                )
                print(f"{step['name']}: {json.dumps(command, ensure_ascii=False)}")
            return 0

        results = run_steps(
            manifest,
            repo_root=args.repo_root,
            profile=profile,
            session=session,
            output_dir=args.output,
            timeout=args.timeout,
        )
        print(json.dumps({"steps": [result.to_dict() for result in results]}, indent=2))
        return 0 if all(result.passed for result in results) else 1
    except CheckpointManifestError as exc:
        return _print_error(exc)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
