from PIL import Image

from . import loader, schema, utils


def make_image(job: schema.StableDiffusionJob) -> Image.Image:
    model = loader.load_stable_diffusion(job.model, job.is_xl)
    model_kwargs = {}

    if job.is_xl:
        model_kwargs["aesthetics_score"] = job.aesthetics_score
        model_kwargs["quality_score"] = job.quality_score

    with utils.timed_scope("Inference"):
        return model.forward(job.width, job.height, **model_kwargs)
