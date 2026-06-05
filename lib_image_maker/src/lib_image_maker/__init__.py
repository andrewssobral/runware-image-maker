import abc
import threading
import weakref
from typing import Any, Self

import PIL.Image

_TOTAL: int = 1024**3 * 32
_FREE: int = _TOTAL
_MUTEX: threading.Lock = threading.Lock()


def memory() -> tuple[int, int]:
    "FREE, TOTAL"
    return _FREE, _TOTAL


def _alloc(n: int) -> None:
    with _MUTEX:
        global _FREE  # noqa: PLW0603
        if n + _FREE < 0:
            raise MemoryError
        elif n + _FREE > _TOTAL:
            raise OverflowError
        _FREE += n


class Model(abc.ABC):
    """Base class for all models.
    Initializes weights on class creation."""

    def __init__(self, data: bytes) -> None:
        _alloc(-self.parameters() * 2)
        weakref.finalize(self, _alloc, self.parameters() * 2)
        self._read_weights(data)
        __import__("time").sleep(self.parameters() / 1024**3 / 4)

    def _forward(self, width: int, height: int, c: float) -> PIL.Image.Image:
        __import__("time").sleep(self.parameters() / 5e9)
        return PIL.Image.new("F", (width, height), color=c).convert("RGB")

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> PIL.Image.Image:  # noqa: ANN401
        "Creates an image from model inputs"

    @abc.abstractmethod
    def parameters(self) -> int:
        "Model parameter count"

    @abc.abstractmethod
    def _read_weights(self, data: bytes) -> Self:
        pass


class SD15Model(Model):
    def forward(self, width: int, height: int) -> PIL.Image.Image:
        return self._forward(width, height, 50)

    def parameters(self) -> int:
        return int(1e9)

    def _read_weights(self, data: bytes) -> Self:
        if not data.decode().startswith("stable-diffusion-v1-5"):
            raise RuntimeError
        return self


class SDXLModel(Model):
    def forward(self, width: int, height: int, aesthetics_score: float, quality_score: float) -> PIL.Image.Image:
        return self._forward(width, height, aesthetics_score * 5 + quality_score / 2)

    def parameters(self) -> int:
        return int(7e9)

    def _read_weights(self, data: bytes) -> Self:
        if "xl" not in data.decode().lower():
            raise RuntimeError
        return self


class Flux1Model(Model):
    def forward(self, width: int, height: int, guidance_embedding: float) -> PIL.Image.Image:
        return self._forward(width, height, guidance_embedding * 10)

    def parameters(self) -> int:
        return int(26e9)

    def _read_weights(self, data: bytes) -> Self:
        if "flux.1" not in data.decode().lower():
            raise RuntimeError
        return self
