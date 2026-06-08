from PIL import Image

import lib_image_maker

from . import loader, schema, utils


def make_image(job: schema.StableDiffusionJob) -> Image.Image:
    model = loader.load_stable_diffusion(job.model)
    model_kwargs = {}

    if isinstance(model, lib_image_maker.SDXLModel):
        model_kwargs["aesthetics_score"] = job.aesthetics_score
        model_kwargs["quality_score"] = job.quality_score
    elif isinstance(model, lib_image_maker.Flux1Model):
        model_kwargs["guidance_embedding"] = job.guidance_embedding

    with utils.timed_scope("Inference"):
        return model.forward(job.width, job.height, **model_kwargs)
