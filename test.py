import io

import httpx
import PIL.Image
import pytest

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


def post(data: dict[str, str | float]) -> PIL.Image.Image:
    return PIL.Image.open(
        io.BytesIO(
            httpx.post(
                "http://localhost:12345/image_maker",
                json=data,
                timeout=60,
            )
            .raise_for_status()
            .read()
        )
    )


@pytest.mark.parametrize(("model"), SD15_MODELS + SDXL_MODELS)
def test_models(model: str) -> None:
    post({"model": model, "is_xl": model in SDXL_MODELS}).verify()


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
    assert post({"model": model, "width": w, "height": h, "is_xl": model in SDXL_MODELS}).size == (w, h)


@pytest.mark.parametrize(
    ("data", "value"),
    [
        ({"model": SD15_MODELS[0]}, 50),
        ({"model": SDXL_MODELS[0], "is_xl": True, "aesthetics_score": 8, "quality_score": 80}, 80),
        ({"model": SDXL_MODELS[-1], "is_xl": True, "aesthetics_score": 5, "quality_score": 90}, 70),
    ],
)
def test_jobs(data: dict[str, float | str], value: float) -> None:
    img = post(data).convert("F")
    flat = img.get_flattened_data()
    avg = sum(flat) / len(flat)  # pyright: ignore
    assert abs(avg - value) < 1e-8
