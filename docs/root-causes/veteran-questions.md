# 🎖️ Veteran Question List

> Questions a CUDA performance expert asks when debugging GPU training.
> Organized by investigation phase — start at the top and drill down.

---

## Phase 1: Orientation

1. What GPU hardware is being used? (H100, A100, etc.)
2. How many GPUs? What parallelism strategy? (DP, TP, PP, FSDP?)
3. What framework? (PyTorch, JAX, Megatron, DeepSpeed, vLLM?)
4. What's the batch size and sequence length?
5. How long is the profile? How many iterations are captured?
6. Was the profile taken during warmup or steady state?

## Phase 2: GPU Utilization

7. What's the GPU utilization percentage? (kernel active time / total time)
8. Are there visible idle gaps ("bubbles") in the GPU timeline?
9. Is the GPU idle time concentrated or distributed across the iteration?
10. Do all streams look busy, or is work concentrated on a single stream?

## Phase 3: Kernel Analysis

11. What are the top 5 kernels by total time?
12. Is any single kernel > 30% of total GPU time?
13. Are there many small kernels (< 10μs)? What's the launch overhead?
14. Are matrix multiplications hitting efficient tile sizes? (multiples of 128)
15. Is FlashAttention being used? What version?

## Phase 4: Memory & Data Movement

16. How much H2D transfer is happening? Is it in the critical path?
17. Is `pin_memory=True` set in the DataLoader?
18. Are there any unexpected D2H transfers? (`.item()`, `.cpu()`, `print(tensor)`)
19. What's the peak GPU memory usage vs. available memory?

## Phase 5: Communication (Multi-GPU)

20. What percentage of iteration time is NCCL?
21. Is NCCL overlapping with compute?
22. Which collective dominates? (AllReduce, AllGather, ReduceScatter?)
23. Are all ranks balanced, or do some finish early?
24. What interconnect? (NVLink, InfiniBand, Ethernet?)

## Phase 6: System Overheads

25. Is there a CPU thread at > 90% utilization?
26. Are there GC pauses visible in the timeline?
27. Is the first iteration significantly slower? (compilation/warmup)
28. Are there `cudaDeviceSynchronize` calls in the profiled window?
29. Is `torch.compile()` being used? Is it effective?

## Phase 7: Root Cause Hypothesis

30. Is this problem **compute-bound**, **memory-bound**, or **communication-bound**?
31. If compute-bound: which kernel? Can we fuse or replace it?
32. If memory-bound: what's the arithmetic intensity? Are we hitting bandwidth limits?
33. If communication-bound: can we overlap better? Reduce message sizes?
34. Is there a **simple fix** (config change) or does this need **code changes**?

## Phase 8: Verification

35. If we fix the hypothesized root cause, what speedup do we expect?
36. Can we isolate the fix and re-profile to confirm?
37. Did the fix introduce any new bottlenecks?
38. What's the MFU before and after?

---

## Meta-Questions (For the Debugging Process)

- Are we profiling the right workload? (Not too short, not warmup-only)
- Is the profile clean? (No profiler overhead artifacts)
- Have we looked at enough iterations to see variance?
- Are we comparing apples to apples? (Same config, same data)
