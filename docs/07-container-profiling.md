# Container Profiling

> **Source:** [Nsight Systems User Guide — Container and Scheduler Support](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#container-and-scheduler-support)

---

## Overview

Nsight Systems works inside containers (Docker, Kubernetes, etc.) with some additional setup. The CLI is strongly recommended over the GUI for container environments.

## Docker Setup

### Enabling `perf_event_open`

Nsight Systems requires the `perf_event_open` syscall for CPU sampling. Three ways to enable:

#### Option 1: Privileged mode
```bash
docker run --privileged=true ...
```

#### Option 2: Add SYS_ADMIN capability
```bash
docker run --cap-add=SYS_ADMIN ...
```

#### Option 3: Custom seccomp profile (most secure)
1. Download the default Docker seccomp profile
2. Add `perf_event_open` to the allowed syscalls:
   ```json
   { "name": "perf_event_open", "action": "SCMP_ACT_ALLOW", "args": [] }
   ```
3. Save as `default_with_perf.json`
4. Launch with:
   ```bash
   docker run --security-opt seccomp=default_with_perf.json ...
   ```

### Example Docker Launch

```bash
# With NVIDIA runtime
sudo nvidia-docker run \
  --network=host \
  --security-opt seccomp=default_with_perf.json \
  --rm -ti \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash

# Inside container:
nsys profile --trace=cuda,nvtx python train.py
```

### Without CPU Sampling (Simpler)

If you only need GPU tracing (no CPU sampling), you can skip the `perf_event_open` setup:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none python train.py
```

## Modal-Specific Notes

Modal containers run on NVIDIA GPU instances. Key considerations:

1. **Use CUDA devel images** — they include `nsys`:
   ```python
   modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04")
   ```

2. **Disable CPU sampling** — Modal containers may not have `perf_event_open`:
   ```bash
   nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none ...
   ```

3. **Use Modal Volumes** for persistent profile storage
4. **Export to SQLite inside the container** — avoid transferring large `.nsys-rep` files

## Kubernetes

Nsight Systems supports profiling in Kubernetes pods:
- Deploy nsys CLI in the application pod
- Use `--cap-add=SYS_ADMIN` in the pod security context
- Use shared volumes between pods for report files

## Nsight Streamer

For long-running services, Nsight Systems provides a streamer that can:
- Stream profiling data to a remote GUI
- Allow remote start/stop of collection
- No need to pre-configure collection duration

## Best Practices for Container Profiling

1. **Install only the CLI** — minimal footprint, no GUI dependencies needed
2. **Share report files** via mounted volumes or cloud storage
3. **Profile inside, analyze outside** — run `nsys profile` in container, `nsys export`/`nsys stats` can be done locally
4. **Test `nsys --version`** first to verify installation
5. **Use `--force-overwrite=true`** to avoid issues with existing files in shared volumes
