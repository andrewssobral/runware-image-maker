import gc
import threading

import lib_image_maker

from . import utils


class ModelManager:
    def __init__(self) -> None:
        self._held: dict[str, lib_image_maker.Model] = {}
        self._mutex: threading.Lock = threading.Lock()

    def store(self, key: str, model: lib_image_maker.Model) -> None:
        with self._mutex:
            self._held[key] = model

    def get(self, key: str) -> lib_image_maker.Model | None:
        with self._mutex:
            return self._held.get(key, None)

    def free(self, num_bytes: int) -> bool:
        with self._mutex:
            free_bytes, _ = lib_image_maker.memory()

            while free_bytes < num_bytes:
                try:
                    dropped: str = next(iter(self._held.keys()))
                except StopIteration:
                    break
                del self._held[dropped]
                gc.collect()
                utils.IM_LOGGER.info(f"Dropped model {dropped}")
                free_bytes, _ = lib_image_maker.memory()

            return free_bytes >= num_bytes


MANAGER: ModelManager = ModelManager()
"Global model manager"
