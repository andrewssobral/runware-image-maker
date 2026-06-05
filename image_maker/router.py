import io

import fastapi

import lib_image_maker

from . import engine, schema, utils

ROUTER: fastapi.APIRouter = fastapi.APIRouter()


@ROUTER.post("/image_maker")
def image_maker(job: schema.StableDiffusionJob) -> fastapi.responses.Response:
    utils.IM_LOGGER.info("Received job")
    with utils.timed_scope("Make Image"):
        img = engine.make_image(job)
    imgio = io.BytesIO()
    img.save(imgio, format="PNG", compress_level=1)
    return fastapi.responses.Response(imgio.getvalue(), media_type="image/png")


@ROUTER.get("/memory")
def memory() -> int:
    return lib_image_maker.memory()[0]
