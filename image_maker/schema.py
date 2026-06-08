from typing import Self

import pydantic

import lib_image_maker

from . import loader


class StableDiffusionJob(pydantic.BaseModel, extra="forbid", use_attribute_docstrings=True):
    "Use an AI model to make images"

    prompt: str = pydantic.Field(min_length=1)
    "Text prompt describing the image to generate"

    model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    "The model to use for inference"

    width: int = pydantic.Field(default=1024, ge=128, le=2048)
    "Image width in pixels"

    height: int = pydantic.Field(default=1024, ge=128, le=2048)
    "Image height in pixels"

    aesthetics_score: float = pydantic.Field(default=2, ge=0, le=10)
    "SDXL only!"
    quality_score: float = pydantic.Field(default=30, ge=0, le=100)
    "SDXL only!"

    guidance_embedding: float | None = pydantic.Field(default=None, ge=0, le=10)
    "Guidance scale — required for Flux.1 models, not valid for others"

    @pydantic.model_validator(mode="after")
    def _validate_family_params(self) -> Self:
        try:
            family = loader.detect_model(self.model)
        except RuntimeError as exc:  # unrecognized model -> clean 422, not a later 500
            raise ValueError(str(exc)) from exc

        is_flux = family is lib_image_maker.Flux1Model
        is_sdxl = family is lib_image_maker.SDXLModel

        if is_flux and self.guidance_embedding is None:
            msg = "guidance_embedding is required for Flux.1 models"
            raise ValueError(msg)
        if not is_flux and self.guidance_embedding is not None:
            msg = "guidance_embedding is only valid for Flux.1 models"
            raise ValueError(msg)

        sdxl_only = {"aesthetics_score", "quality_score"} & self.model_fields_set
        if not is_sdxl and sdxl_only:
            msg = f"{', '.join(sorted(sdxl_only))} are only valid for SDXL models"
            raise ValueError(msg)

        return self
