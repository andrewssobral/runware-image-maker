import gc
from collections.abc import Iterator

import pytest

import lib_image_maker
from image_maker import memory

_GIB = 1024**3


@pytest.fixture
def manager() -> Iterator[memory.ModelManager]:
    memory.MANAGER._held.clear()  # reclaim the shared pool from prior in-process loads (test.py)
    gc.collect()
    mgr = memory.ModelManager()
    yield mgr
    mgr._held.clear()  # release any held models so the process-global pool resets
    gc.collect()


def test_free_returns_true_when_pool_has_room(manager: memory.ModelManager) -> None:
    # Regression for the FREE/USED inversion: on an empty 32 GiB pool, ensuring 6 GiB
    # free must succeed (the old bug returned False here).
    assert manager.free(6 * _GIB) is True


def test_free_returns_false_when_request_exceeds_pool(manager: memory.ModelManager) -> None:
    _, total = lib_image_maker.memory()
    assert manager.free(total + _GIB) is False  # unsatisfiable, but must not raise


def test_get_missing_returns_none(manager: memory.ModelManager) -> None:
    assert manager.get("absent") is None


def test_get_refreshes_lru_recency(manager: memory.ModelManager) -> None:
    for key in ("a", "b", "c"):
        manager.store(key, object())
    manager.get("a")  # bump 'a' to most-recently-used
    assert list(manager._held) == ["b", "c", "a"]
    assert next(iter(manager._held)) == "b"  # eviction target is the LRU entry


def test_store_places_new_key_at_mru_end(manager: memory.ModelManager) -> None:
    manager.store("a", object())
    manager.store("b", object())
    assert list(manager._held) == ["a", "b"]


def test_free_evicts_least_recently_used(manager: memory.ModelManager) -> None:
    a, b = "stabilityai/stable-diffusion-xl-base-1.0", "Lykon/dreamshaper-xl-1-0"
    manager.store(a, lib_image_maker.SDXLModel(a.encode()))  # ~13 GiB
    manager.store(b, lib_image_maker.SDXLModel(b.encode()))  # ~13 GiB
    manager.get(a)  # bump 'a' -> 'b' is now least-recently-used

    assert manager.free(10 * _GIB) is True  # forces evicting one model
    assert manager.get(b) is None  # the LRU model was evicted
    assert manager.get(a) is not None  # the MRU model was kept
