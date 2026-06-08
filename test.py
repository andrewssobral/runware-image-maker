import io

import fastapi
import PIL.Image
import pytest
from fastapi.testclient import TestClient

from image_maker import router

SD15_MODELS: list[str] = [
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
]

SDXL_MODELS: list[str] = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "Lykon/dreamshaper-xl-1-0",
    "Lykon/dreamshaper-xl-lightning",
    "Lykon/dreamshaper-xl-v2-turbo",
    "cagliostrolab/animagine-xl-4.0",
]

FLUX_MODELS: list[str] = [
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
]


_app = fastapi.FastAPI()
_app.include_router(router.ROUTER)
_CLIENT = TestClient(_app)


def post(data: dict[str, str | float]) -> PIL.Image.Image:
    data = {"prompt": "a scenic landscape", **data}
    response = _CLIENT.post("/image_maker", json=data)
    response.raise_for_status()
    return PIL.Image.open(io.BytesIO(response.content))


@pytest.mark.parametrize(("model"), SD15_MODELS + SDXL_MODELS)
def test_models(model: str) -> None:
    post({"model": model}).verify()


@pytest.mark.parametrize(
    ("model", "w", "h"),
    [
        (SD15_MODELS[0], 512, 512),
        (SD15_MODELS[0], 768, 512),
        (SD15_MODELS[0], 256, 640),
        (SDXL_MODELS[0], 1280, 832),
    ],
)
def test_resolutions(model: str, w: int, h: int) -> None:
    assert post({"model": model, "width": w, "height": h}).size == (w, h)


@pytest.mark.parametrize(
    ("data", "value"),
    [
        ({"model": SD15_MODELS[0]}, 50),
        ({"model": SDXL_MODELS[0], "aesthetics_score": 8, "quality_score": 80}, 80),
        ({"model": SDXL_MODELS[-1], "aesthetics_score": 5, "quality_score": 90}, 70),
        ({"model": FLUX_MODELS[0], "guidance_embedding": 3.5}, 35),
        ({"model": FLUX_MODELS[1], "guidance_embedding": 3.5}, 35),
    ],
)
def test_jobs(data: dict[str, float | str], value: float) -> None:
    img = post(data).convert("F")
    flat = img.get_flattened_data()
    avg = sum(flat) / len(flat)  # pyright: ignore
    assert abs(avg - value) < 1e-8


@pytest.mark.parametrize(
    "data",
    [
        {"model": FLUX_MODELS[0]},  # Flux requires guidance_embedding
        {"model": SDXL_MODELS[0], "guidance_embedding": 5},  # guidance is Flux-only
        {"model": SD15_MODELS[0], "guidance_embedding": 5},  # guidance is Flux-only
        {"model": SD15_MODELS[0], "aesthetics_score": 8},  # scores are SDXL-only
        {"model": "acme/unknown-model", "guidance_embedding": 1},  # unrecognized model
    ],
)
def test_invalid_family_params_rejected(data: dict[str, str | float]) -> None:
    assert _CLIENT.post("/image_maker", json={"prompt": "x", **data}).status_code == 422
