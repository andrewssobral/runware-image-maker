import threading
from collections.abc import Callable

import lib_image_maker

from . import memory, utils

AnyModel = lib_image_maker.SD15Model | lib_image_maker.SDXLModel | lib_image_maker.Flux1Model


class ModelTooLargeError(Exception):
    "The model cannot fit the VRAM pool even at its configured precision (permanent)."


class InsufficientVRAMError(Exception):
    "VRAM was momentarily exhausted; the load can be retried (transient)."


# Ordered string->class registry for zero-allocation family detection. Each predicate
# mirrors the corresponding model's `_read_weights` in lib_image_maker. Predicates can
# overlap (first match wins), so the most specific is checked first: Flux's `flux.1`
# before SDXL's broad `xl` substring — otherwise a Flux path containing "xl" would be
# misdetected as SDXL. Keep in sync if the lib's accepted model strings change.
_REGISTRY: list[tuple[type[lib_image_maker.Model], Callable[[str], bool]]] = [
    (lib_image_maker.SD15Model, lambda s: s.startswith("stable-diffusion-v1-5")),
    (lib_image_maker.Flux1Model, lambda s: "flux.1" in s.lower()),
    (lib_image_maker.SDXLModel, lambda s: "xl" in s.lower()),
]


def detect_model(url: str) -> type[lib_image_maker.Model]:
    "Return the model class for a model string without constructing (no VRAM alloc)."
    for cls, matches in _REGISTRY:
        if matches(url):
            return cls
    msg = f"Unrecognized model: {url}"
    raise RuntimeError(msg)


# Server-side precision policy (A3): Flux loads at 4-bit so it fits the 32 GiB pool
# (fp16 would need 48.43 GiB); SD15/SDXL stay fp16. bytes_per_param() is the single
# source of truth — it sizes free() AND is passed to the lib's load reserve, so the
# freed amount always matches what the load consumes (C9 watch-item). int8 (1.0) is
# the named conservative fallback (A2) if 4-bit ever needs revisiting.
_DEFAULT_BYTES_PER_PARAM: float = 2.0  # fp16
_PRECISION: dict[type[lib_image_maker.Model], float] = {
    lib_image_maker.Flux1Model: 0.5,  # 4-bit
}


def bytes_per_param(model_cls: type[lib_image_maker.Model]) -> float:
    "Load precision (bytes/param) for a model family — the server policy (A3)."
    return _PRECISION.get(model_cls, _DEFAULT_BYTES_PER_PARAM)


def required_bytes(model_cls: type[lib_image_maker.Model]) -> int:
    "VRAM a model reserves on load; reads parameters() without constructing (no alloc)."
    return int(object.__new__(model_cls).parameters() * bytes_per_param(model_cls))


# Serializes the cold-load section so concurrent requests for the same model don't
# double-allocate / double-load. A single global lock (per-model locks are deferred,
# C12); the warm cache-hit path never acquires it, so inference stays parallel.
_LOAD_LOCK: threading.Lock = threading.Lock()


def load_stable_diffusion(url: str) -> AnyModel:
    # Warm path: cache hit needs no lock.
    if (model := memory.MANAGER.get(url)) is not None:
        return model

    # Cold path: serialize, then double-check (another thread may have loaded it
    # while we waited on the lock).
    with _LOAD_LOCK:
        if (model := memory.MANAGER.get(url)) is None:
            with utils.timed_scope(f"Loaded model {url}"):
                model_cls = detect_model(url)
                bpp = bytes_per_param(model_cls)
                need = required_bytes(model_cls)

                _, total = lib_image_maker.memory()
                if need > total:  # can never fit — fail before evicting anything
                    need_gib, total_gib = need / 1024**3, total / 1024**3
                    msg = f"{model_cls.__name__} needs {need_gib:.1f} GiB > {total_gib:.0f} GiB pool"
                    raise ModelTooLargeError(msg)

                if not memory.MANAGER.free(need):
                    utils.IM_LOGGER.warning("Did not free enough memory for the model, inference may fail!")

                try:
                    model = model_cls(url.encode(), bytes_per_param=bpp)
                except MemoryError:
                    memory.MANAGER.free(total)  # drop all cached models (free the whole pool) and retry once
                    try:
                        model = model_cls(url.encode(), bytes_per_param=bpp)
                    except MemoryError as exc:
                        msg = "Insufficient VRAM to load the model; please retry"
                        raise InsufficientVRAMError(msg) from exc

                memory.MANAGER.store(url, model)

    assert isinstance(model, AnyModel), f"Model was unknown type {type(model)}"
    return model
