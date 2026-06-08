# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A FastAPI image-generation service. `lib_image_maker` is a *mock* inference
library: it simulates VRAM allocation and fakes "inference" via `time.sleep`
proportional to parameter count, returning a flat-color PIL image. There is no
real GPU or model weights — `image_maker/memory.py` byte accounting and timings
are all driven by the simulation in `lib_image_maker/src/lib_image_maker/__init__.py`.

## Commands

- Install deps: `uv sync`
- Run the server: `uv run main.py` (serves on `http://localhost:12345`)
- Run tests: `uv run pytest` — runs the whole suite **standalone, no server needed**
  (`test.py` drives the app in-process via FastAPI `TestClient`; `test_memory.py` is
  server-free unit tests). `uv run main.py` is only for actually serving.
- Run a single file/test: `uv run pytest test_memory.py` / `uv run pytest -k test_resolutions`
- Lint: `uvx ruff check` / format: `uvx ruff format` (ruff isn't a project dep; `EXE002`
  on every file is a harmless artifact of the external-drive mount, not a code issue)
- Watch simulated VRAM: `uv run monitor.py` (polls `/memory` every 0.5s)

## Architecture

Request flow: `router.py` (`POST /image_maker`) → `engine.make_image` →
`loader.load_stable_diffusion` → `lib_image_maker` model `.forward()` → PNG response.

- **schema.py** — `StableDiffusionJob` pydantic request model (`extra="forbid"`):
  `prompt` (required, `min_length=1`), `model`, `width`/`height`, SDXL
  `aesthetics_score`/`quality_score`, and Flux `guidance_embedding`. A
  `model_validator` derives the model family from `model` (via `loader.detect_model`)
  and enforces per-family required/forbidden fields.
- **loader.py** — `detect_model` infers the model class (SD15/SDXL/Flux1) from the
  model string via a zero-allocation string→class registry (most-specific predicate
  first). `load_stable_diffusion` serializes the cold-load path under a single global
  lock (double-checked), sizes eviction from the real footprint
  (`parameters() × bytes_per_param`), loads Flux at 4-bit per the precision policy,
  and maps capacity failures to `ModelTooLargeError` (→400) / `InsufficientVRAMError`
  (→503).
- **memory.py** — global `MANAGER` (`ModelManager`): a mutex-guarded **LRU** model
  cache keyed by model string. `free()` evicts least-recently-used models
  (`OrderedDict` + `move_to_end` on `get()`) until enough simulated VRAM is free.
- **lib_image_maker** — model classes (`SD15Model`, `SDXLModel`, `Flux1Model`)
  subclass `Model`. `Model.__init__` takes an optional `bytes_per_param` precision
  (default `2.0` = fp16); each validates the model string in `_read_weights` and has a
  fixed `parameters()` count that drives both VRAM use and sleep duration.

## Conventions

- Python 3.12, fully type-annotated (ruff `ANN` enforced). Line length 120.
- Use `utils.timed_scope(label)` to log timed sections; log via `utils.IM_LOGGER`
  (no `print` — ruff `T20`).
- `lib_image_maker` is a forked workspace member updated independently of the
  server; treat its API as an upstream boundary. (The one change made there — an
  optional `bytes_per_param` load precision — is additive and backward-compatible.)

## Development guidelines

Be cautious and surgical when changing this codebase.

- **Before changing anything**, understand the existing architecture; state the root
  cause, the proposed fix, the files affected, and any risks — and get confirmation
  before implementing.
- **Keep changes minimal and surgical.** Solve only the requested problem; don't
  refactor unrelated code, add abstractions, or introduce dependencies without need.
  Prefer the smallest safe change that fully resolves the issue.
- **Preserve behaviour and contracts.** Assume existing behaviour is intentional.
  Don't change public APIs or interfaces without approval; if a change is breaking,
  stop and explain what breaks, why it's necessary, and the non-breaking alternatives.
- **Validate.** Confirm the fix addresses the issue and nothing else; check for
  regressions; keep existing tests green; run the relevant tests. Don't modify tests
  unless the change requires it.
- **Commit in small, focused units** — one logical change each (one fix / feature /
  improvement), never mixing unrelated changes. Use clear, descriptive messages in
  conventional-commit style (e.g. `fix(memory): correct FREE/USED inversion`,
  `feat(api): add required prompt field`).
- **When multiple solutions exist**, present the alternatives and trade-offs,
  recommend the least invasive, and confirm before implementing.

Default mindset: *the smallest change that safely solves the problem while preserving
existing behaviour.*

## Project docs (read these first)

These docs are the source of truth — keep them in sync, don't duplicate their content here:

- **README.md** — the *goal* (integrate Flux.1, improve performance) and the
  CONTRIBUTING rules (document assumptions instead of asking; existing tests must
  pass; disclose AI usage).
- **docs/FINDINGS.md** — the *diagnosis*: verified bugs and design gaps with
  file:line references and reproducible repros (captured against the baseline).
- **docs/DESIGN.md** — the *redesign*: API / memory / testing design, the rollout,
  and the **§5 decision log** with rationale (reconciled post-implementation).
- **docs/SELF_REVIEW.md** — the *change summary*: what shipped, mapped to FINDINGS.
- **CLAUDE.md** (this file) — the *map*: architecture, commands, conventions.

(Ignore other files under `docs/` unless explicitly asked.)

## Status

Flux.1 is integrated (`FLUX.1-dev` and `FLUX.1-schnell`, loaded at 4-bit to fit the
32 GiB pool) and all five FINDINGS are fixed: `free()` FREE/USED + FIFO→LRU,
registry-based detection, `is_xl` removed in favour of server-side family inference,
`guidance_embedding` wired, and a 400/503 capacity error contract. See
`docs/SELF_REVIEW.md` for the change summary and `docs/DESIGN.md §5` for the decision
log behind each choice.
