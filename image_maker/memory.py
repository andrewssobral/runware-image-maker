import gc
import threading
from collections import OrderedDict

import lib_image_maker

from . import utils


class ModelManager:
    def __init__(self) -> None:
        self._held: OrderedDict[str, lib_image_maker.Model] = OrderedDict()
        self._mutex: threading.Lock = threading.Lock()

    def store(self, key: str, model: lib_image_maker.Model) -> None:
        with self._mutex:
            self._held[key] = model

    def get(self, key: str) -> lib_image_maker.Model | None:
        with self._mutex:
            model = self._held.get(key)
            if model is not None:
                self._held.move_to_end(key)  # mark most-recently-used
            return model

    def free(self, num_bytes: int) -> bool:
        with self._mutex:
            free_bytes, _ = lib_image_maker.memory()

            while free_bytes < num_bytes:
                try:
                    dropped: str = next(iter(self._held.keys()))
                except StopIteration:
                    break
                del self._held[dropped]
                # Force immediate reclaim of the model's simulated VRAM. On CPython this is
                # redundant (refcount-zero on `del` fires weakref.finalize at once), but we keep
                # it as defensive insurance: lib_image_maker is an independently-updated upstream
                # boundary that could introduce reference cycles, and on non-refcounting runtimes
                # (e.g. PyPy) del does NOT promptly finalize — without this, the memory() re-read
                # below would not see the freed bytes. Eviction is rare, so the cost is negligible.
                gc.collect()
                utils.IM_LOGGER.info(f"Dropped model {dropped}")
                free_bytes, _ = lib_image_maker.memory()

            return free_bytes >= num_bytes


MANAGER: ModelManager = ModelManager()
"Global model manager"
