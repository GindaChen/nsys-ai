# 📖 The Book of Root Causes

> A living document of GPU performance problems, their signatures, and their solutions.
> Each entry represents lessons learned from real-world debugging — many costing millions in GPU hours.

---

## 1. GPU Bubbles (Pipeline Stalls)

**Symptom:** Idle gaps between consecutive kernel executions on a stream.

**Why it happens:**
- CPU is too slow to enqueue the next kernel (Python overhead, data preprocessing)
- Synchronization barriers (`cudaDeviceSynchronize`, `torch.cuda.synchronize()`) force the GPU to drain
- Data not ready (waiting for H2D transfer or NCCL to complete)

**How to detect:**
```bash
nsys-ai skill run gpu_idle_gaps profile.sqlite
```
Look for gaps > 1ms between kernels on the same stream.

**How to fix:**
- Use CUDA graphs to pre-record kernel sequences
- Overlap data loading with compute using multiple streams
- Replace explicit syncs with event-based dependencies
- Use `torch.compile()` to fuse small kernels

**Real-world example:** A Megatron-LM training run showed 15% idle time because the data loader was synchronizing on every micro-batch. Moving to async prefetching eliminated the bubbles entirely.

---

## 2. CPU Bottleneck

**Symptom:** GPU utilization is < 80% even though there's work to do. CPU threads show high utilization.

**Why it happens:**
- Python GIL contention between data loading and model execution
- Complex preprocessing on CPU (tokenization, augmentation)
- Eager-mode PyTorch has per-op Python overhead
- Excessive logging or metric computation on the critical path

**How to detect:**
```bash
nsys-ai skill run thread_utilization profile.sqlite
```
Look for a single thread at > 90% CPU usage.

**How to fix:**
- Move preprocessing to GPU or a separate process
- Use `torch.compile()` to reduce per-op Python overhead
- Reduce logging frequency in training loops
- Use `torch.utils.data.DataLoader` with `num_workers > 0` and `pin_memory=True`

---

## 3. NCCL Serialization

**Symptom:** AllReduce/AllGather operations run sequentially instead of overlapping with compute.

**Why it happens:**
- Gradient bucketing is misconfigured (DDP bucket sizes too large or too small)
- NCCL streams are blocked by compute on the same GPU stream
- Network topology is suboptimal (cross-node collectives over slow interconnect)

**How to detect:**
```bash
nsys-ai skill run nccl_breakdown profile.sqlite
```
Total NCCL time > 20% of iteration time is a red flag.

**How to fix:**
- Tune DDP bucket sizes: `torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=25)`
- Ensure NCCL runs on a separate stream (default in PyTorch DDP)
- Use gradient compression or `FSDP` for large models
- Check NVLink/InfiniBand topology matches the collective algorithm

---

## 4. Excessive Host-to-Device Transfers

**Symptom:** Large H2D memory copies appear in the critical path, creating gaps.

**Why it happens:**
- Data tensors are created on CPU and moved per-batch instead of pre-staged
- Loss computation pulls tensors to CPU for logging
- Weights or buffers are repeatedly moved to GPU

**How to detect:**
```bash
nsys-ai skill run memory_transfers profile.sqlite
```
H2D total time > 5% of iteration time is concerning.

**How to fix:**
- Use `pin_memory=True` in DataLoader for async H2D
- Keep all model parameters on GPU (check for accidental CPU tensors)
- Accumulate metrics on GPU, sync to CPU only at checkpoint time
- Pre-allocate GPU buffers instead of re-creating tensors each iteration

---

## 5. Small Kernel Overhead

**Symptom:** Profile shows thousands of tiny kernels (< 10μs each) with high launch overhead.

**Why it happens:**
- Eager-mode PyTorch launches one kernel per op
- Element-wise operations that could be fused (e.g., `x + bias` followed by `relu`)
- Custom CUDA kernels that don't batch work

**How to detect:**
```bash
nsys-ai skill run kernel_launch_overhead profile.sqlite
```
Overhead > kernel duration is a strong signal.

**How to fix:**
- Use `torch.compile()` to fuse element-wise ops
- Enable `torch.backends.cudnn.benchmark = True` for conv layers
- Use CUDA graphs for static computation sequences
- Write custom fused CUDA/Triton kernels for hot paths

---

## 6. Kernel Hotspot

**Symptom:** A single kernel accounts for > 50% of total GPU time.

**Why it happens:**
- Flash Attention kernel on very long sequences
- Matrix multiplication on an under-optimized shape
- Custom kernel that hasn't been tuned

**How to detect:**
```bash
nsys-ai skill run top_kernels profile.sqlite
```
The first kernel in the list dominates.

**How to fix:**
- For matmul: ensure shapes are multiples of 128 (H100) or 64 (A100)
- For attention: use FlashAttention v2/v3, check sequence length padding
- Profile the specific kernel to determine if it's compute-bound or memory-bound
- Check SM occupancy and warp utilization with NCU

---

## 7. GC Pauses

**Symptom:** Periodic idle gaps in GPU timeline, correlating with Python garbage collection.

**Why it happens:**
- PyTorch creates many small temporary tensors that trigger GC
- Training loops generate circular references (autograd graph)
- Default GC thresholds are too low for ML workloads

**How to detect:**
Look for regular, periodic gaps in `gpu_idle_gaps` that don't correlate with batch boundaries.

**How to fix:**
- Disable GC during training: `gc.disable()`, manual `gc.collect()` between iterations
- Use `torch.no_grad()` during evaluation to reduce tensor graph complexity
- Reduce tensor creation in the loop (reuse buffers via `output.copy_()`)

---

## 8. Compute-Communication Imbalance

**Symptom:** Some ranks finish compute early and idle at NCCL barriers. Uneven NCCL wait times across ranks.

**Why it happens:**
- Data parallel: some micro-batches have more tokens/pixels (variable sequence length)
- Pipeline parallel: stage imbalance (one stage has more layers/compute)
- Tensor parallel: split doesn't divide evenly across SMs

**How to detect:**
Compare `nccl_breakdown` across multiple rank profiles. Look for asymmetric wait times.

**How to fix:**
- Pad or bucket sequences to uniform length
- Rebalance pipeline stages (move layers between stages)
- Use dynamic micro-batch sizing

---

## 9. Module Loading in Hot Path

**Symptom:** First iteration is 10-100× slower than steady state.

**Why it happens:**
- Python `import` statements inside the forward pass
- `torch.compile()` JIT compilation on first call
- Triton kernel compilation on first invocation
- cuDNN autotuning

**How to detect:**
Look at timestamps in `gpu_idle_gaps` — if the first major gap is at iteration start + the gap is never repeated → it's one-time compilation.

**How to fix:**
- Call `torch.compile()` outside the training loop with a warmup input
- Pre-compile Triton kernels: `@triton.jit(warmup=True)`
- Run one warmup iteration before starting the profiled window
- Use `--trim` to exclude warmup from analysis

---

## 10. Excessive Synchronization

**Symptom:** `cudaDeviceSynchronize` appearing frequently in CUDA Runtime trace.

**Why it happens:**
- Explicit sync calls in user code
- `.item()`, `.cpu()`, or `print(tensor)` in the training loop
- Error checking code that forces synchronization
- Some PyTorch ops silently synchronize

**How to detect:**
```bash
nsys-ai skill run kernel_launch_overhead profile.sqlite
```
Also search for `cudaDeviceSynchronize` in CUDA Runtime events.

**How to fix:**
- Remove `.item()` from the training loop — accumulate on GPU
- Use `torch.cuda.set_sync_debug_mode(1)` to detect hidden syncs
- Replace `print(loss.item())` with `print(loss)` (prints without sync)
- Use async error checking (`CUDA_LAUNCH_BLOCKING=0`)

---

## Contributing

This document grows with every debugging session. To add a new root cause:

1. Describe the **symptom** (what does the profile look like?)
2. Explain the **mechanism** (why does this happen?)
3. Show the **detection method** (which skill or query?)
4. Provide the **fix** (what did you actually do?)
5. Include a **real-world example** if possible
