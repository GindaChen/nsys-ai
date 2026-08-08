# scripts/

Developer utilities for local workflows. Some scripts under this directory are used by CI
(see `.github/workflows/plugin-smoke.yml` for `smoke_test.sh` and `build_fixture.py`).
The CUDA axpy workload below is not.

## CUDA axpy workload (`axpy.cu`)

A tiny CUDA program that launches the same `axpy` kernel N times. It exists so a developer
with a GPU can produce a real nsys capture and exercise `nsys-ai profile`, `diff`, and related
paths against genuine GPU activity — the kind of coverage issue #289 measured as missing
(committed fixtures are small; CI has no GPU).

Source only is committed. Do not check in the binary or any capture. This workload is for a
developer machine with `nvcc`, a CUDA GPU, and `nsys` on `PATH`. It is not wired into CI.

### Build

```bash
nvcc -O2 -o /tmp/axpy scripts/axpy.cu
/tmp/axpy 5
```

Example output:

```text
axpy done: launches=5 n=1048576
```

Launch count defaults to 5 when omitted; pass an integer to control how many kernels run.

### Capture a before/after pair

`-o` / `--output` is a fresh artifact **directory**. The SQLite export lands at
`<dir>/profile.sqlite`. A path like `<dir>.sqlite` does not exist and fails with
`PROFILE_NOT_FOUND`.

With the package installed (`pip install -e '.[dev]'`, per the README):

```bash
mkdir -p /tmp/axpy-captures

nsys-ai profile -o /tmp/axpy-captures/before -- /tmp/axpy 5
nsys-ai profile -o /tmp/axpy-captures/after  -- /tmp/axpy 40
```

Example capture output (5 launches):

```text
[preparing] 0.0s
[capturing] 0.0s
[exporting] 2.2s
[validating] 2.2s
[finished] 2.2s
Report: /tmp/axpy-captures/before/profile.nsys-rep
SQLite: /tmp/axpy-captures/before/profile.sqlite
RunSpec: /tmp/axpy-captures/before/runspec.json
Profile ID: nsys2:sha256:<content hash of your capture>
Export schema: 3.25.0
Nsight version: <whatever your nsys reports>
Kernels: 5
```

A successful capture prints `Kernels: <n>` matching the launch count (here `Kernels: 5`,
then `Kernels: 40`).

Wrong path (sibling `.sqlite` instead of `<dir>/profile.sqlite`):

```bash
nsys-ai diff \
  /tmp/axpy-captures/before/profile.sqlite \
  /tmp/axpy-captures/after/profile.sqlite
```

```text
Error [PROFILE_NOT_FOUND]: profile not found: /tmp/axpy-captures/before/profile.sqlite
```

Correct compare:

```bash
nsys-ai diff \
  /tmp/axpy-captures/before/profile.sqlite \
  /tmp/axpy-captures/after/profile.sqlite
```

Example kernel regression line:

```text
 +706.54us  |         5->40 | axpy
```

Varying the launch count on one binary is enough to get a measurable kernel-time diff
(for example 5 vs 40 launches).
