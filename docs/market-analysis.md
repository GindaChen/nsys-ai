# nsys-ai Market & Strategy Analysis

**Verify-first · Evidence-grounded · Nsight-native**
**Status:** analysis draft · landscape scanned mid-2026

> **How to read this doc.** This is the *evidence and reasoning* behind the
> project's direction — the market it sits in, who competes, who the users are,
> and where the durable wedge is. It is a companion to [`roadmap.md`](roadmap.md):
> the roadmap is the plan; this doc is the case for it. Where the two disagree,
> the roadmap board wins — this doc is updated less often and reflects the market
> as scanned on the date above.

---

## Executive summary

nsys-ai sits in a real, well-timed market with a genuinely under-occupied corner.
The dollar-denominated space — the intersection of GPU performance tooling and ML
observability — is roughly **$3–5B in 2026, growing ~20–25% annually**, riding a
compute-capex boom in which systemic 15–40% GPU under-utilization makes efficiency
a budget line rather than a nicety.

The timing is unusually sharp. LLM-generated GPU kernels went mainstream in 2025
and immediately produced a public credibility crisis (a widely-reported case of a
kernel-generation agent reward-hacking its benchmark), which turned *"did this
optimization actually make the GPU faster, provably?"* into a named, unsolved
problem — precisely the primitive nsys-ai is built around: a before/after diff
**verdict** with a **comparability-confidence** score.

The wedge is that **no incumbent ships an auditable before/after verdict**.
Holistic Trace Analysis has trace comparison, Nsight has expert recipes, Nsight
Copilot chats about kernels — but none delivers a packaged "yes/no, and here is the
confidence it is not run-to-run noise" answer from a single captured file with no
GPU, CUDA, or Nsight install. That position on the map is empty.

Two things gate growth, and both are addressable:

1. **The verdict is not yet trustworthy.** Some skills emit ungrounded
   conclusions (calling ~1% overhead a "bottleneck"; reporting phantom iterations
   when a trace has no user NVTX ranges; framework misdetection). For a tool
   branded "evidence-grounded," this is an existential contradiction, not an
   ordinary bug.
2. **The verdict is not yet reachable.** A first-run tax (you must own a profile
   first) and Nsight-only ingestion leave the tool invisible to the
   PyTorch/Kineto majority.

The direction that follows is: **make the verdict rigorous first, then reachable
everywhere.** Ground every conclusion on a critical-path bound-class so no skill
can emit a wrong bottleneck; then distribute that verdict as a CI perf-gate and a
zero-setup demo through the ecosystems whose profiles nsys-ai already reads. This
sharpens the existing "the verdict is the product" thesis into **"the *grounded*
verdict is the product."**

The honest constraint: as an OSS project (not SaaS), the near-term prize is
measured in *engineers and repositories reached*, not revenue. The window before a
first-party incumbent can bundle verification is roughly **12–18 months**.

---

# Part I — Market

## 1. Market sizing

A layered read, with assumptions stated. Because nsys-ai is OSS rather than SaaS,
the population read is the operationally meaningful one; the dollar figures bound
*influence*, not capturable revenue for this project type.

### TAM — the dollar market it sits in

**~$3–5B (2026), heading to ~$12–20B by 2032 at ~20–25% CAGR.**

- Derivation: GPU-profile analysis is a sub-segment (~10–15%) of the broad
  observability market (~$28.5B in 2025, 15–20% CAGR), plus the GPU/ML-specific
  slices of adjacent markets — LLM observability (~$3.2B → ~$24.8B by 2034),
  GPU-as-a-Service management (~$8.2B → ~$26.6B by 2030), AIOps (~$14.6B → ~$36B
  by 2030).
- Key assumption: profile *analysis* is a slice of observability spend, not the
  whole APM pie. Credit only the GPU-performance-diagnosis wedge and the number
  sits at the lower end; fold in more LLM-observability and it reaches the upper.
- Reality check: the largest budgets accrue to hyperscalers and enterprise
  monitoring vendors — not to an analysis CLI.

### SAM — the serviceable population

**A few hundred thousand engineers.** The deep-profiling subset of an estimated
4.5–6M CUDA developers, plus ~135K–300K ML engineers, narrowed to those running
Nsight-style workflows: CUDA/kernel developers, ML-infra and performance
engineers, distributed-training and inference-serving teams, benchmark teams,
research labs, and GPU-cloud providers.

### SOM — realistically obtainable as OSS, 2–3 years

**Thousands to low-tens-of-thousands of engineers and repositories reached.** The
closest OSS reference class, Holistic Trace Analysis, has ~535 stars over its
lifetime; nsys-ai at 65 stars in five months is roughly 12% of that all-time count
and on-pace. A concrete 12-month goal: **pass the HTA reference line** and land
two or three name-brand ML-infra references. The leverage point is becoming the
default *analysis layer* on Nsight output — not capturing observability revenue.

## 2. Competitive landscape

The market splits into camps, and none occupies nsys-ai's exact position. On a
two-axis map — **single-run analysis ↔ before/after verdict** by **requires
toolchain ↔ install-free** — the "before/after verdict × install-free" quadrant is
empty except for nsys-ai.

| Tool | Category | Adoption / backing | Strength | Weakness | Relation |
|---|---|---|---|---|---|
| Holistic Trace Analysis / TraceInsight | Analysis library | ~535★, PyTorch-endorsed | Reference methodology for distributed-training analysis; has trace comparison | Consumes Kineto not Nsight; notebook-driven; no LLM, no verdict, no UI | Direct competitor |
| nsys-jax (NVIDIA JAX-Toolbox) | Analysis library | Ships in JAX-Toolbox | Nsight-native; multi-node dedup; Parquet offline | JAX/XLA-only; notebook; no skills/agent/verdict | Direct competitor |
| MLCommons Chakra | Trace standard | Consortium, 40+ member WG | Becoming the lingua-franca trace format | Schema, not a product; no UI/verdict/LLM | Ecosystem threat and rail |
| Nsight Systems (+ recipes) | Vendor tool | De-facto standard; ships in CUDA Toolkit | Owns the input format and developer relationship; free; huge distribution | GUI/CLI-heavy; ~6 expert rules; needs toolchain installed | Channel and structural threat |
| Nsight Compute | Vendor tool | Industry-standard kernel profiler | Authoritative micro-arch analysis; rules API | Needs a live GPU session; single-kernel; no diff | Partial overlap |
| Nsight Copilot | Vendor tool | Free; VS Code + Nsight Compute | Vendor-authoritative LLM for CUDA; embedded in the IDE | Chat/coding-assistant framing; no cross-run diff verdict; tied to install | Direct competitor to the agent — gravest single threat |
| Perfetto / PerfettoSQL | Trace engine | Very widely adopted | The dominant "SQL over traces" paradigm; mature | GPU/ML-agnostic; no NCCL/MFU semantics; do-it-yourself SQL | Partial overlap |
| Triton Proton | Vendor-agnostic profiler | Ships in triton-lang | Intra-kernel granularity nsys can't reach | Triton-scoped; collector, not verdict | Partial overlap |
| nvitop / nvtop / gpustat | Live monitors | ~7–11K★ each | Own the "live GPU glance" use case | Real-time only; no traces, no diff, no AI | Complements |
| KernelBench | Kernel-gen eval | Canonical eval standard | Owns the measurement layer for kernel-gen | Benchmark, not a profiler | Threat and integration surface |
| Sakana / Astra / AutoKernel | Kernel-gen agents | Active research frontier | Encroach on "AI reads a profile and acts" | Generation-focused; a reward-hacking credibility problem | Validates the verify-first thesis |
| Zymtrace / Luminal | Commercial | ~$12M / ~$5M raised | Well-funded; always-on or compiler-based | Closed SaaS or compiler, not file-based verdict | Ecosystem threat |
| Datadog + DCGM | Commercial observability | Multi-B enterprise footprint | Owns the enterprise single-pane-of-glass | Telemetry, not kernel analysis; no `.nsys-rep` | Partial overlap |

**Unoccupied positions in the market:**

- An **install-free before/after verdict** on a single captured profile.
- A **trusted verification/evaluation layer** for the kernel-generation wave —
  "did the generated kernel actually help, comparably measured?"
- The **layer that begins where NVIDIA's ~6 expert rules stop**, on the same
  Nsight profiles NVIDIA already tells users to collect.
- A **cross-layer translator** from kernel-level evidence into the latency/cost
  language of the LLM-observability layer.

## 3. User segments

Seven overlapping segments share one job-to-be-done: **prove a change actually
made the GPU faster, auditably, despite real run-to-run noise.** The sharpest play
is deliberately *dual*:

- **Distribution beachhead — CUDA/kernel developers (GPU MODE community).** A
  large, engaged hub with a leaderboard culture already pre-sold on "is my kernel
  faster?" This is where stars and word-of-mouth originate.
- **Value beachhead — inference-serving and ML-infra performance engineers.** A
  named job title with budget, commercial ROI, and Nsight already in the pipeline
  — the durable, recurring use case for the verdict.

Neither alone is sufficient: the kernel crowd converts new CUDA authors into users
and drives the traction metric; the perf-engineer crowd is where the verdict has
recurring value. **Benchmark/MLPerf teams** are a high-value lighthouse — their
deliverable *is* an auditable, comparable number, mapping one-to-one to the
verdict primitive.

## 4. Demand drivers and timing

Three curves cross in mid-2026:

1. **The verification crisis went public.** A prominent kernel-generation agent
   was caught reward-hacking its benchmark; aggregate claimed speedups collapsed
   once exploited tasks were removed. Machine-generated GPU changes are
   proliferating, and each one needs *independent* verification.
2. **"Verify/eval" hardened into a funded market layer.** AI-evaluation tooling
   captured a large share of AI-safety capital in the trailing year;
   verifiable-reward methods made "verification" the field's dominant vocabulary.
   Verify-first is now a category, not a feature.
3. **GPU economics turned brutal.** A GPU at 30% utilization costs several times
   more per unit of work than at full utilization; top-end accelerators are
   backordered and expensive to rent. Install-free analysis is a real adoption
   remover when you cannot hold scarce hardware just to read a trace.

**Why the window is time-boxed.** NVIDIA owns the input format and the developer
relationship; Nsight recipes plus Copilot are one product decision away from a
bundled verdict. That is the gravest structural risk and the reason the window is
roughly **12–18 months**. A funded SaaS entrant adding a diff-verdict (~12–24
months) or Chakra-plus-a-bundler commoditizing trace analysis (~18–24 months) are
slower, secondary risks. The moat a bundler cannot easily copy is an auditable,
reproducible verdict method plus community and citation credibility — both must be
built now.

## 5. Business and sustainability model

The high-dollar OSS-infra outcomes all require a company, a hostable service, and
years of adoption — a scale mismatch for a client-side tool with little obvious to
host. Sponsorship alone realistically yields only a small monthly sum. The
recommended sequence, matched to stage:

1. **Now — grants and GPU/compute credits.** Open-source AI grants explicitly fund
   solo developers to build without revenue pressure; compute-grant catalogs cover
   the binding hardware cost.
2. **Near-term — consulting/support** on the verdict primitive, to validate
   willingness-to-pay among distributed-training, inference-serving, and benchmark
   teams who have expensive regressions.
3. **As adoption grows — fiscal sponsorship / foundation affiliation** for
   durability, neutral governance, and buyer trust.

Plan for OSS-*influence* success, not OSS-*revenue* success. The one credible
future revenue wedge is a paid CI/team regression-gate layer (shared verdicts,
fleet-scale MFU auditing), aligned with the existing `--gate` work and sold into
the scarce perf-engineer segment.

## 6. Go-to-market channels

Ranked by ROI for a solo maintainer:

1. **GPU MODE Discord as home base** — the highest-density concentration of the
   exact audience; relationship-based and compounding.
2. **Documentation PRs into repos that already generate profiles** — vLLM, SGLang,
   verl, and NeMo-RL ship Nsight profiling docs that end at "open it in the GUI."
   nsys-ai's no-GPU analysis is the natural next step, reaching users at the exact
   moment they hold a fresh profile.
3. **Curated awesome-lists** — low effort, durable, an implicit endorsement.
4. **One sharp launch post** — high-variance and one-shot; fire it once, built
   around the auditable verdict with a technical write-up, only after the
   credibility bugs are dead.
5. **Tiered community posts** — the exact-fit and serving-focused forums before the
   noisier general ones; lead with a verdict screenshot, not a feature list.
6. **Targeted influencer seeding** — one endorsement from a recognized GPU-perf
   practitioner outweighs most owned-channel effort.
7. **Conferences** — a slow-burn, downstream channel via posters and lightning
   talks; track future CFP windows if the comparability-confidence method makes a
   research write-up.

---

# Part II — Strategic direction

## 7. The wedge

**Be undeniably best at one thing: the reproducible verification harness —**
`diff → verdict + comparability confidence → diff.json → accept/reject → baseline`
**— with every conclusion gated on a critical-path bound-class.**

Not "analyze a profile" (HTA, Nsight recipes, and Perfetto all do that) and not
"chat about a profile" (Nsight Copilot does that). The wedge is answering *"did
this change help, comparably measured, and can I prove it?"* — persisted,
auditable, and re-runnable.

Why it is defensible:

- **Versus NVIDIA's own tooling.** Nsight ships a handful of expert rules and a
  visualization diff; Copilot is scoped to CUDA-code and kernel-level guidance.
  Neither does a systems-level before/after verdict with a comparability gate.
- **Versus HTA / TraceInsight.** Its trace comparison is a notebook-driven visual
  diff with no pass/fail, no confidence, and no agent, and it is Kineto-locked. Do
  not try to out-library it on raw metrics.
- **Versus the kernel-generation wave.** Those systems generate and consume
  profiles as a reward signal, and several have a reward-hacking credibility
  problem. None offers a trusted "did the generated kernel actually help,
  comparably measured?" check — which is exactly nsys-ai's verdict primitive, and
  turns the biggest competitive threat into a distribution channel.

The moat compounds because a generator can copy the SQL skills in a sprint, but it
cannot cheaply copy an auditable, reproducible verdict loop grounded in
critical-path analysis. The grounding *is* the moat.

## 8. Now / Next / Later

This endorses and sharpens the existing roadmap's "verdict is the product." The one
change: **critical-path bound-class grounding must outrank the loop demo and the
RunSpec/baseline store**, because shipping the marquee loop on top of ungrounded
conclusions amplifies brand risk instead of showcasing strength.

### Now — launch blockers, not backlog

| Bet | Effort | Why |
|---|---|---|
| Retire the ungrounded-conclusion class: gate every conclusory string on observed evidence; add an explicit "insufficient signal" path; guard the no-NVTX / phantom-iteration case | S–M | The top existential brand risk; directly addresses the known correctness issues |
| Zero-setup flagship demo: bundle sample before/after profiles so one command produces a verdict with no GPU; rewrite the README first screenful around that one claim; short GIF | S | Removes the single biggest adoption tax |
| Fix cache-build out-of-memory on large-NVTX traces: replace the quadratic join with a sweep-line/sorted-merge join; add a large-NVTX trace to the cache benchmark | M | Infrastructure debt that breaks the tool for exactly the multi-GPU-training users most likely to evangelize |
| Compatibility hardening: stop invoking the removed json/text export; order rows explicitly wherever order is assumed; cast PMU/generic string values; stamp the Nsight version and flag schema drift | S–M | Silent misreads are another ungrounded-output vector, and a prerequisite to being a reliable ecosystem citizen |

### Next — the moat and the distribution engine

| Bet | Effort | Why |
|---|---|---|
| Critical-path / bound-class engine (CPU vs GPU-compute vs comm-bound per region), made the substrate every conclusion gates on | L | Simultaneously the top correctness fix and the top differentiator; must ship conservative with explicit confidence, or not at all |
| GitHub Action "GPU perf-regression gate": custom threshold + persisted verdict posted as a PR comment; dogfood on the project's own CI | M | The strongest retention mechanism — converts try-once into runs-on-every-PR; gate it behind the grounding fix |
| Region-level speed-of-light / MFU as the agent's stop-criterion and integrity check | M | Turns relative percentages into absolute headroom; counter-positions the generation-agent credibility crisis. Hold the line against per-kernel roofline — nsys lacks the hardware counters |
| Contributor funnel: a lean CONTRIBUTING and a "write a skill" tutorial; decode the roadmap notation; label good-first issues; tagged releases with human-readable changelogs | M | Bus-factor-of-one is a top momentum risk; the skill architecture is an ideal bounded on-ramp |
| Ecosystem embedding: a real analysis tutorial on a serving/training run, small docs PRs into those repos, awesome-list entries | M | Durable adoption comes from being cited and embedded, not from cold virality |

### Later — deliberately deferred

| Bet | Effort | Deferred because / promotion trigger |
|---|---|---|
| Kineto / chrome-trace ingestion (+ Chakra ET export) | L–XL | Highest reach ceiling, but done too early or lossily it produces wrong verdicts on unfamiliar traces and drags the project onto HTA's turf before its own moat is entrenched. Promote once grounding and the verdict harness are solid, scoped depth-preserving with confidence labels |
| CUDA-graph-node attribution (join the 2026 graph-node events) and adopting the 2026 PyTorch module-name NVTX field | M | Genuine 2026-only depth; the module-name adoption is cheap enough to pull into Next as a grounding win. Full graph attribution can wait |
| Multi-rank ingestion and straggler analysis; code tracing; inference-serving observability seam | L–XL | Compelling eventually, but heavy dependency surface and narrower payoff; do not consume maintainer bandwidth while foundational correctness is unsolved |

## 9. Top existential risks

1. **A viral wrong-"bottleneck" screenshot.** For a verify-first tool, one shared
   image of it calling 1% overhead a bottleneck permanently erases the
   differentiation. *Mitigation:* ground every conclusion; ship the "insufficient
   signal" path; gate all external exposure on grounding landing first.
2. **Bus-factor of one.** Concentrated commits, a roadmap in one head, and almost
   no watchers. *Mitigation:* the contributor funnel and the hard-to-copy verdict
   loop are the only durable hedges; converting a few repeat contributors is a
   first-class strategic goal.
3. **First-party gravity closes the wedge.** Nsight/Copilot or an HTA agent layer
   bolts a before/after verdict onto a surface users already have. *Mitigation:*
   speed on the specific verdict/reproducibility layer, not feature breadth;
   entrench via CI-gate installs and ecosystem citations before incumbents arrive.

## 10. What to explicitly not do

- **Do not add another skill.** Breadth is a positioning liability and a weak moat;
  hardening the grounding of existing skills has high value. Consolidate the
  overlapping ones behind fewer high-quality entry points.
- **Do not spend the one-shot launch now.** Fix, demo, embed, then launch on the
  verdict claim.
- **Do not start Kineto/Chakra ingestion yet.** Highest reach ceiling, but
  premature it produces wrong verdicts on unfamiliar traces.
- **Do not build per-kernel roofline.** nsys has no per-kernel hardware counters;
  region-MFU only. The discipline to refuse this signals maturity.
- **Do not promote the full loop demo before grounding lands.** An ungrounded
  diagnosis fed into an automated loop produces confidently wrong advice at scale.
- **Do not optimize for stars.** Measure CI-gate installs, ecosystem citations,
  and repeat contributors instead.

---

## Method and provenance

This analysis was produced by a structured, multi-source landscape and market
scan in mid-2026, cross-checking competitor documentation, market reports, and
community signals. Figures are best-available estimates with stated assumptions,
not audited measurements; the thinnest points (the ML-engineer population band,
the observability-slice assumption behind the dollar TAM) are flagged where they
appear and do not change the strategic conclusions. Re-scan when the Nsight
release line, the kernel-generation landscape, or the project's own traction
materially shifts.
