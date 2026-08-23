# Real-project checkpoints

Checkpoint manifests make a profile run auditable. They are deliberately
metadata-first: the repository stores the recipe and checksum, not a large
`.nsys-rep` capture or model weights.

## Validate the contract fixture

From the repository root:

```bash
python3 scripts/checkpoint.py validate \
  examples/checkpoints/b0-contract/manifest.json \
  --repo-root .
```

To verify the committed fixture bytes as well:

```bash
python3 scripts/checkpoint.py validate \
  examples/checkpoints/b0-contract/manifest.json \
  --repo-root . \
  --require-profile
```

Print the exact analysis commands without executing them:

```bash
python3 scripts/checkpoint.py plan \
  examples/checkpoints/b0-contract/manifest.json \
  --repo-root .
```

Run the five analysis steps and write per-step stdout/stderr logs outside the
repository:

```bash
python3 scripts/checkpoint.py run \
  examples/checkpoints/b0-contract/manifest.json \
  --repo-root . \
  --output /tmp/nsys-ai-checkpoint-b0
```

## Moving from B0 to a real project

Copy the manifest shape for vLLM or SGLang and replace every contract fixture
value with a pinned project revision, model/artifact revision, workload trace,
capture environment, and profile checksum. Use `--profile` to point at a local
capture during a manual checkpoint; do not commit the capture itself.

The five analysis steps are intentionally explicit:

```text
doctor → diagnose → ask → diff → review
```

An exit code of zero is not sufficient for adoption. The `expected_signals`
entries must name an independently checkable observation, and a missing
RunSpec, provider, or worker profile must be recorded as an abstention/stop.

The first external recipe after this contract fixture is vLLM single-GPU
offline latency. SGLang follows with its own recipe. Ray is a separate ladder:
Core actors/tasks, Compiled Graph, Serve, Train, and Data are not one checkbox.
