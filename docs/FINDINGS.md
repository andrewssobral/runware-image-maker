# Findings

*A diagnosis of the codebase as I first found it (baseline commit `90482fb`), written
before I changed anything. Everything described here has since been fixed — see
`SELF_REVIEW.md` for the resolutions and `DESIGN.md` for the redesign. File references
point to the original code.*

## How the system works

A FastAPI service turns a request into a PNG — but the "GPU" is entirely simulated. A
library called `lib_image_maker` tracks a 32 GiB VRAM pool in a module global, has each
model reserve `parameters × 2` bytes when it loads, and fakes inference with a `sleep`
proportional to the model's size. Nothing touches a real GPU or real weights. For
debugging that's a gift: every memory and timing behaviour is deterministic, so the bugs
below reproduce reliably rather than being hardware flakiness.

The task has two stated goals — integrate Flux.1, and improve performance — and both run
straight into the defects below.

## The bugs

Five of them, each verified by running the code.

**1. The cache's eviction logic is inverted.** The memory manager's `free()` asks the
simulator how much memory is available, then unpacks the answer backwards — it reads the
*free* figure into a variable it treats as *used*. Every comparison after that is
inverted. The symptom is stark: on a completely empty 32 GiB GPU, asking to free 6 GiB
reports failure. The service works at all only because a fallback path catches the
resulting error and bulk-evicts the *entire* cache — so the careful, incremental
eviction the code appears to perform never actually runs. (`memory.py`)

**2. Flux.1 cannot fit in memory at all.** Flux is 26 billion parameters; at 2 bytes
each that's about 48 GiB, against a 32 GiB pool. This isn't a tuning problem — even a
perfectly empty GPU is 16 GiB too small, so no amount of eviction can help. The README
says both Flux models "should work on our current GPUs," but at full precision that
isn't true. Making Flux fit means changing how much memory it uses — loading it at lower
precision — which is a change inside `lib_image_maker`, not just the server. This is the
core of the task. (`lib_image_maker`)

**3. Loading Flux crashes with a 500.** To decide which kind of model a request wants,
the loader tries constructing each candidate class until one accepts the name — but
constructing a model *reserves its VRAM up front*, before it even validates the name. So
identifying a Flux model means trying to allocate its ~48 GiB, which fails; the single
retry hits the same wall, and that second failure isn't caught, so it escapes as an HTTP
500. (`loader.py`)

**4. Even if Flux loaded, inference would crash.** Flux's `forward()` requires a
`guidance_embedding` argument, but the engine only ever passes the SDXL-specific
arguments, and only when a client flag is set. There isn't even a `guidance_embedding`
field in the request schema to supply it with. (`engine.py`, `schema.py`)

**5. Dispatch trusts a client flag over reality.** Which arguments inference receives is
decided by an `is_xl` boolean the client sets by hand — separate from the model the
server actually loaded. The two can disagree (mark an SD15 model as XL and you'll send
SDXL arguments to an SD15 model), and the flag is redundant anyway, since the server
already knows the family from the model name. (`engine.py`, `schema.py`)

## Performance gaps

Beyond outright bugs, a few things work against the performance goal:

- **Detection allocates memory just to throw it away.** Because the loader identifies a
  model by constructing candidates (see bug 3), every wrong guess reserves and releases
  real VRAM — and under memory pressure those throwaway allocations can trip eviction on
  their own.
- **The cache evicts oldest-first, not least-used.** A model loaded early is the first
  to go, even if it's the most requested — the worst policy for a serving cache.
- **Cold loads can stampede.** Two requests for the same not-yet-loaded model can both
  miss the cache and both load it — twice the memory, twice the latency — because nothing
  holds a lock across the load.
- **Every model's first request pays the full load time** on the request path, since
  nothing is loaded ahead of time.

## The redesign

How I addressed all of this — the API, the memory manager, type detection, the Flux-fit
decision, concurrency, and testing — is in `DESIGN.md`, along with the reasoning behind
each choice. This document is just the diagnosis.
