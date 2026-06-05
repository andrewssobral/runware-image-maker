import lib_image_maker

from . import memory, utils

AnyModel = lib_image_maker.SD15Model | lib_image_maker.SDXLModel | lib_image_maker.Flux1Model


def auto_model(data: bytes, _n: int = 0, _last_error: BaseException | None = None) -> AnyModel:
    try:
        return AnyModel.__args__[_n](data)
    except IndexError:
        if _last_error:
            raise _last_error
        else:
            raise
    except Exception as e:  # noqa: BLE001 # It gets re-raised later if it fails.
        return auto_model(data, _n + 1, e)


def load_stable_diffusion(url: str, is_xl: bool) -> AnyModel:
    if (model := memory.MANAGER.get(url)) is None:
        with utils.timed_scope(f"Loaded model {url}"):
            if is_xl:
                freed = memory.MANAGER.free(6 * 1024**3)
            else:
                freed = memory.MANAGER.free(1 * 1024**3)

            if not freed:
                utils.IM_LOGGER.warning("Did not free enough memory for the model, inference may fail!")

            try:
                model = auto_model(url.encode())
            except MemoryError:
                memory.MANAGER.free(999 * 1024**3)  # drop all lib_image_maker and try again
                model = auto_model(url.encode())

            memory.MANAGER.store(url, model)

    assert isinstance(model, AnyModel), f"Model was unknown type {type(model)}"
    return model
