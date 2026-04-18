# Quick Start — nsys-ai Claude Skill

## 1. Install nsys-ai

```bash
pip install "nsys-ai[agent]"
```

## 2. Install this skill

```bash
git clone https://github.com/GindaChen/nsys-ai ~/.claude/skills/nsys-ai
```

## 3. Profile your workload

```bash
# Full run
nsys profile python train.py

# Recommended: capture only the training loop
nsys profile --capture-range=cudaProfilerApi python train.py
# (requires torch.cuda.profiler.start()/stop() in your script)
```

This produces `report.nsys-rep`. The skill auto-converts it — no manual export needed.

## 4. Run the skill

In Claude Code, type:
```
/analyze
```

Or with a direct question (skips the mode menu):
```
/analyze report.nsys-rep why is my training slow?
```

## 5. What you get

Within 3 turns:
- Root cause identified (e.g. "NCCL serialization at 18% overlap")
- Code fix with example (e.g. `DDP(model, bucket_cap_mb=256)`)
- Interactive timeline with the bottleneck highlighted

## Common invocations

```
/analyze report.nsys-rep
/analyze before.nsys-rep after.nsys-rep regression
/analyze profile.sqlite what is my mfu?
/analyze profile.sqlite why is flash attention slow?
```
