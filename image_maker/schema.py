import pydantic


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
