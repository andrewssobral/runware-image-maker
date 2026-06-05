import contextlib
import logging
import sys
import time
from collections.abc import Generator

IM_LOGGER: logging.Logger = logging.getLogger("image_maker")
IM_LOGGER.addHandler(logging.StreamHandler(sys.stdout))


@contextlib.contextmanager
def timed_scope(label: str) -> Generator[None, None, None]:
    clock = time.perf_counter()
    yield
    IM_LOGGER.info(f"{label} completed in {time.perf_counter() - clock:.2f} sec")  # noqa: G004
