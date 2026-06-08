# Design — Image Maker

**Author:** Andrews Cordolino Sobral

This is the design write-up the project asks contributors for: *if I could redesign this
system, what would I change?* It covers the redesign behind the Flux.1 integration and
the fixes that made it possible. `FINDINGS.md` is the diagnosis it builds on, and
`SELF_REVIEW.md` summarises what actually shipped.

## The system as it was

A FastAPI service turns a request into a PNG, backed by a *mock* GPU: the
`lib_image_maker` library simulates a 32 GiB VRAM pool, has each model reserve
`parameters × 2` bytes on load, and fakes inference with a `sleep`. A request flows from
the router to an engine, which loads (or reuses a cached) model and calls its `forward()`.

The diagnosis in `FINDINGS.md` turned up five defects — an inverted eviction check, a
type-detection scheme that allocated memory just to guess a model's class, a Flux model
that couldn't fit in the pool at all, missing request fields, and dispatch driven by a
client flag that could contradict the loaded model — plus a few performance gaps. The
redesign below tackles them together, because several are entangled: you can't cleanly
wire Flux through the API without also reworking how the family is detected and how memory
is sized.

## The redesign

**Request API.** The client `is_xl` flag is gone. The server already knows the model
family from the model name, so it derives it once and uses that everywhere — there's no
longer a second, client-supplied "what model is this?" that can disagree with reality. I
added the fields the API was missing: a required `prompt` (the defining input of any image
API, even though the mock ignores its value) and Flux's `guidance_embedding`. A single
validator checks that each family gets exactly the fields it should — Flux requires
guidance, SDXL takes its aesthetic and quality scores, and neither accepts the other's.
Capacity failures now map to honest HTTP codes — **400** when a model can never fit, **503**
(with `Retry-After`) when memory is only temporarily full — instead of leaking an internal
error as a 500.

**Memory and caching.** I fixed the inverted `free()` so it compares against actual free
memory and re-checks after each eviction, and switched the cache from FIFO to LRU so a
frequently used model survives. Eviction is now sized from a model's real footprint — its
parameter count times its load precision — rather than the old fixed 6-or-1 GiB guess, and
that same size feeds an up-front capacity check: a model larger than the whole pool is
rejected immediately rather than discovered through a failed allocation.

**Type detection.** Instead of constructing each candidate model to see which one accepts
the name — which reserved real VRAM on every wrong guess — a small lookup table maps a name
to its class with no allocation. Construction (and its load `sleep`) now happens once, for
the right class only.

**Making Flux fit.** This was the crux. At full precision Flux needs ~48 GiB and the pool
is 32 GiB, so it can't be cached away into fitting — the memory cost itself has to change. I
added an optional precision setting to the model loader and load Flux at 4-bit, which brings
it from ~48 GiB down to ~12 GiB. The default precision is unchanged, so every other model
behaves exactly as before. This is the one change inside the "upstream" `lib_image_maker`
library, and I kept it additive and backward-compatible: the load reserve and its matching
release use the same figure, so the accounting can't drift, and parameter counts are left
alone, so the timing simulation is unaffected.

**Concurrency.** The endpoints run in a thread pool, so requests really do overlap. In this
mock, memory is consumed only at *load* time — inference just sleeps — so I let inference run
concurrently and serialise only the load section, under a single lock with a double-check
inside. That closes the "two requests load the same cold model twice" race while leaving the
common warm path lock-free.

**Testing.** The original tests required a server started by hand in another terminal. I
moved them to FastAPI's in-process `TestClient`, so `uv run pytest` now runs the whole suite
standalone, and added direct unit tests for the memory manager (where the worst bug lived)
and end-to-end coverage for both Flux models.

## Decisions worth calling out

A few choices had real alternatives; these are the ones I'd expect a reviewer to ask about.

**4-bit vs int8 for Flux.** Both fit — int8 lands around 24 GiB, 4-bit around 12. I chose
4-bit because it leaves enough headroom for Flux and another model to stay resident at once,
which matters for the "GPU utilisation" goal. The honest caveat: the mock has no notion of
image quality, but in reality 4-bit costs more quality than int8, so in production I'd
revisit this against a quality bar — int8 is the conservative fallback.

**Precision is the server's call, not the client's.** I deliberately did *not* expose
precision as a request field. It's an infrastructure detail, not a creative parameter, and
exposing it would re-introduce exactly the kind of client-asserted "fact" I just removed with
`is_xl` (a client could ask for full-precision Flux and get a capacity error for their
trouble). If per-request precision ever became a real need, I'd add it as a constrained
choice — int8 or 4-bit only — not a raw number.

**A validator, not a discriminated union (yet).** Pydantic can model each family as its own
request type in a discriminated union, which is cleaner in principle. But the natural
discriminator here is the model name, and wiring that into a union is more machinery than
three families justify. A single validator that derives the family and checks each one's
fields is the least-disruptive change that fully removes `is_xl`; the union is the right
end-state if the API grows more families.

**One global load lock, not per-model locks.** Per-model locks would let two *different* cold
models load in parallel. But cold loads are rare once the cache is warm, and a single lock is
much simpler and still fixes the duplicate-load race. I left per-model locking as a future
optimisation rather than build it speculatively.

**Keeping a redundant `gc.collect()`.** On CPython, deleting a model frees its simulated VRAM
immediately, so the `gc.collect()` in the eviction loop is technically redundant — a
reviewer's tool flagged it. I kept it anyway, with a comment explaining why: it's cheap
(eviction is rare), and it keeps the loop correct if the upstream library ever introduces
reference cycles, or on a Python runtime without reference counting where `del` doesn't free
right away.

## How the work was staged

To keep each step small and reviewable, the changes landed as a sequence of commits in a
dependency-respecting order: memory correctness first (the eviction fix, then LRU), then the
detection lookup table that unlocks family inference and exact sizing, then concurrency, then
the API changes (removing `is_xl`, adding `prompt`), then Flux bottom-up — the library
precision setting, the loader's 4-bit policy, and the schema and engine wiring — followed by
the error contract and finally the new tests. Each commit keeps the suite green, and the
sequence is visible as five reviewed pull requests in the repository (linked from
`SELF_REVIEW.md`).

## AI usage

Per the project's disclosure request:

- **Harness and model:** Claude Code (Anthropic), Claude Opus 4.8.
- **Approach:** I used two Claude Code instances in deliberately separate roles — one
  read-write instance writing and testing the code, and one read-only instance for diagnosis,
  code review, and keeping a running record of design decisions and their rationale. The
  read-only instance made no source changes. Separating the two keeps an audit trail of *why*
  each choice was made and gives every change an independent read before it lands — closer to
  a two-person review than a single agent editing unsupervised.
- **Process:** read-only diagnosis (`FINDINGS.md`) → this design write-up → staged
  implementation, each step reviewed → this self-review.
- **Representative prompts** (paraphrased, not verbatim): *"Verify your suspicions about the
  memory manager and Flux empirically, read-only — report bugs with file references and
  reproductions."*; *"Write the redesign design doc — API, memory, type detection, the
  Flux-fit blocker, concurrency, testing."*; *"Lay out the options and trade-offs for loading
  Flux at reduced precision and recommend one."*; *"Fix `free()` (correct the FREE/USED
  inversion, switch to LRU) and add unit tests for the manager."*; *"Review this diff for
  correctness, edge cases, and convention drift, and confirm the tests pass."*

AI assistance was substantial across diagnosis, design, and implementation; the final
decisions were mine.
