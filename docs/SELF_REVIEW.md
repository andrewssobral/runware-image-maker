# Self-Review

**Author:** Andrews Cordolino Sobral

A summary of the work to integrate Flux.1 into the image service and fix the defects
that were blocking it. Two companion documents go deeper: `FINDINGS.md` (the problems
I found, with reproductions) and `DESIGN.md` (how I approached the redesign and the
trade-offs behind each decision).

## The short version

The service runs on a *mock* GPU — a simulated 32 GiB VRAM pool, with "inference"
faked by a `sleep`. Underneath it were five real defects, from an inverted eviction
check to a model that simply could not fit in memory. I fixed all five, integrated
both Flux.1 models end to end, and tightened the request API and the model cache along
the way. The changes landed as a series of small, self-contained commits, and the test
suite — now a single `uv run pytest` with no server to start first — is green at 27
tests.

## What was wrong, and what I changed

**The memory manager was quietly broken.** Its `free()` routine read the simulator's
`(free, total)` pair in the wrong order, so every eviction decision was inverted: on a
completely empty GPU it would still refuse to make room. The system only limped along
because a fallback path bulk-evicted *everything* whenever a load failed. I corrected
the inversion, and while I was there switched the cache from FIFO to LRU so that a
frequently used model isn't the first one dropped.

**Model type was detected by trial and error — expensively.** To identify a model, the
loader constructed each candidate class in turn, and construction *reserved VRAM*.
Wrong guesses allocated (and later released) gigabytes before being rejected, and for
Flux this "allocate just to check the type" step actually ran the pool out of memory
and surfaced as an opaque HTTP 500. I replaced it with a small lookup table that maps a
model name to its class with no allocation at all.

**Flux didn't fit — and this was the heart of the task.** At full precision, Flux.1
needs about 48 GiB, but the pool is 32 GiB, so no amount of cache eviction could ever
make room; the README's claim that it "should work on our current GPUs" isn't true as
written. The honest fix is to load it at lower precision. I added an optional precision
setting to the model loader — defaulting to the original behaviour, so nothing else
changes — and load Flux at 4-bit, which brings it down to about 12 GiB. That's small
enough to sit in memory alongside the other models rather than evicting them. This was
the one place I had to modify the upstream library, so I kept the change additive and
backward-compatible.

**The request contract trusted the client too much.** Which arguments inference
received was driven by an `is_xl` boolean the client set by hand — and it could
disagree with the actual model, producing the wrong arguments and a runtime error. I
removed it and let the server infer the model family from the name, validating each
family's fields in one place. I also added the fields the API was missing — a required
`prompt` and Flux's `guidance_embedding` — and gave capacity failures honest HTTP
responses: **400** when a model can never fit, **503** when memory is only momentarily
full, instead of a generic 500.

## Breaking changes

Two, both intentional and flagged here for the reviewer:

- `is_xl` has been removed — requests that still send it are now rejected.
- `prompt` is now required.

The existing tests were updated to match.

## Assumptions I made

Following the project's "document your assumptions rather than ask" guidance:

- **Flux loads at 4-bit** to fit the 32 GiB budget. The mock has no notion of image
  quality, but in a real system 4-bit trades some quality for memory; **int8** (~24
  GiB) is the conservative fallback if quality matters more than keeping other models
  resident at the same time.
- **Precision is a server-side policy, not a client option** — it's an infrastructure
  detail, not a creative parameter, so the API doesn't expose it.

## Testing

The suite now runs standalone with `uv run pytest` — `test.py` drives the app
in-process through FastAPI's `TestClient`, so there's no server to start first, and a
new `test_memory.py` unit-tests the memory manager directly (including a regression
test for the inverted-eviction bug). I added end-to-end coverage for both `FLUX.1-dev`
and `FLUX.1-schnell`, and checks that the API rejects invalid field combinations. All
27 tests pass; the ~50-second runtime is almost entirely the mock's deliberate
`sleep`-based "inference," not real work.

## Browse the changes

The work landed as five small, reviewed pull requests, each keeping the test suite green:

1. **[Memory fixes](https://github.com/andrewssobral/runware-image-maker/pull/1)** —
   correct the inverted `free()` and switch the cache from FIFO to LRU.
2. **[Loader core](https://github.com/andrewssobral/runware-image-maker/pull/2)** —
   replace trial-construction with a lookup table and make the cold-load path atomic.
3. **[API](https://github.com/andrewssobral/runware-image-maker/pull/3)** — remove
   `is_xl`, add a required `prompt`, and validate fields per model family.
4. **[Flux](https://github.com/andrewssobral/runware-image-maker/pull/4)** — add the
   4-bit precision load path so both Flux models run end to end.
5. **[Errors & tests](https://github.com/andrewssobral/runware-image-maker/pull/5)** —
   the 400/503 capacity contract, the standalone test suite, and Flux coverage.

## What I'd do next

A few things I deliberately kept out of scope, in rough priority order: per-model load
locks so two *different* cold models can load in parallel; optional warm-up of
frequently used models at start-up; and, if the API grows more model families, moving
the request schema to a proper per-family discriminated union.

## How this was built

This work was done with Claude Code (Opus 4.8), using two instances in distinct roles
— one writing and testing the code, and one read-only for diagnosis and review. The
full AI-usage disclosure, including representative prompts, is in `DESIGN.md`.
