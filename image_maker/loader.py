import threading
from collections.abc import Callable

import lib_image_maker

from . import memory, utils

AnyModel = lib_image_maker.SD15Model | lib_image_maker.SDXLModel | lib_image_maker.Flux1Model

# Ordered string->class registry for zero-allocation family detection. Each predicate
# mirrors the corresponding model's `_read_weights` in lib_image_maker; order matches
# AnyModel and must be preserved (predicates can overlap — first match wins). Keep in
# sync if the lib's accepted model strings change.
_REGISTRY: list[tuple[type[lib_image_maker.Model], Callable[[str], bool]]] = [
    (lib_image_maker.SD15Model, lambda s: s.startswith("stable-diffusion-v1-5")),
    (lib_image_maker.SDXLModel, lambda s: "xl" in s.lower()),
    (lib_image_maker.Flux1Model, lambda s: "flux.1" in s.lower()),
]


def detect_model(url: str) -> type[lib_image_maker.Model]:
    "Return the model class for a model string without constructing (no VRAM alloc)."
    for cls, matches in _REGISTRY:
        if matches(url):
            return cls
    msg = f"Unrecognized model: {url}"
    raise RuntimeError(msg)


# Serializes the cold-load section so concurrent requests for the same model don't
# double-allocate / double-load. A single global lock (per-model locks are deferred,
# C12); the warm cache-hit path never acquires it, so inference stays parallel.
_LOAD_LOCK: threading.Lock = threading.Lock()


def load_stable_diffusion(url: str, is_xl: bool) -> AnyModel:
    # Warm path: cache hit needs no lock.
    if (model := memory.MANAGER.get(url)) is not None:
        return model

    # Cold path: serialize, then double-check (another thread may have loaded it
    # while we waited on the lock).
    with _LOAD_LOCK:
        if (model := memory.MANAGER.get(url)) is None:
            with utils.timed_scope(f"Loaded model {url}"):
                if is_xl:
                    freed = memory.MANAGER.free(6 * 1024**3)
                else:
                    freed = memory.MANAGER.free(1 * 1024**3)

                if not freed:
                    utils.IM_LOGGER.warning("Did not free enough memory for the model, inference may fail!")

                model_cls = detect_model(url)
                try:
                    model = model_cls(url.encode())
                except MemoryError:
                    memory.MANAGER.free(999 * 1024**3)  # drop all lib_image_maker and try again
                    model = model_cls(url.encode())

                memory.MANAGER.store(url, model)

    assert isinstance(model, AnyModel), f"Model was unknown type {type(model)}"
    return model
