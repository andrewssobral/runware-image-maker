import io

import fastapi

import lib_image_maker

from . import engine, loader, schema, utils

ROUTER: fastapi.APIRouter = fastapi.APIRouter()


@ROUTER.post("/image_maker")
def image_maker(job: schema.StableDiffusionJob) -> fastapi.responses.Response:
    utils.IM_LOGGER.info("Received job")
    try:
        with utils.timed_scope("Make Image"):
            img = engine.make_image(job)
    except loader.ModelTooLargeError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except loader.InsufficientVRAMError as exc:
        raise fastapi.HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"}) from exc
    imgio = io.BytesIO()
    img.save(imgio, format="PNG", compress_level=1)
    return fastapi.responses.Response(imgio.getvalue(), media_type="image/png")


@ROUTER.get("/memory")
def memory() -> int:
    return lib_image_maker.memory()[0]
