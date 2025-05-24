import os
import base64

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

# You will need to define INPUT_SCHEMA in a 'schemas.py' file
# or define it directly here.
# Add the following to your INPUT_SCHEMA for LoRA support:
# 'loras': {
#     'type': 'array',
#     'required': False,
#     'default': [],
#     'items': {
#         'type': 'object',
#         'properties': {
#             'model_id': {'type': 'string', 'required': True}, # Hugging Face model ID or path to LoRA
#             'weight': {'type': 'number', 'required': True}    # Weight for the LoRA
#         },
#         'required': ['model_id', 'weight']
#     }
# }
from schemas import INPUT_SCHEMA # Assuming INPUT_SCHEMA is defined in schemas.py

torch.cuda.empty_cache()


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.loaded_job_lora_adapters = [] # Keep track of job-specific LoRAs
        self.load_models()

    def load_base(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True, # Assuming VAE is pre-cached
        )

        # New checkpoint URL
        checkpoint_url = "https://huggingface.co/CuteBlueEyed/GeminiX/resolve/main/Gemini_ILMixV5.safetensors"
        
        # NOTE: For from_single_file with local_files_only=True,
        # the content of checkpoint_url must be pre-cached by Hugging Face's system
        # such that the URL itself acts as the cache key, or checkpoint_url
        # should be a local file path to the pre-downloaded .safetensors file.
        # If this model needs to be downloaded on first run, set local_files_only=False
        # or ensure your RunPod environment pre-downloads and caches it.
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_url,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True, # Good practice, though often inferred
            local_files_only=True, # Adhering to the original script's caching strategy
            # Add other necessary SDXL config parameters if not inferred correctly.
        ).to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Refiner Pipeline from cache using identifier
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()

    def cleanup_job_loras(self):
        """Unloads LoRAs that were loaded for a specific job."""
        if self.base and self.loaded_job_lora_adapters:
            # Disable currently set adapters first
            self.base.set_adapters([], adapter_weights=[])
            # Unload the specific adapters
            self.base.unload_lora_weights(*self.loaded_job_lora_adapters)
            print(f"[ModelHandler] Unloaded job LoRAs: {self.loaded_job_lora_adapters}")
            self.loaded_job_lora_adapters = []


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    import json, pprint # For logging

    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    # -------------------------------------------------------------------------
    # LoRA Handling
    # -------------------------------------------------------------------------
    base_pipe = MODELS.base
    
    # Clean up LoRAs from any previous job on this worker
    MODELS.cleanup_job_loras()

    loras_to_load = job_input.get("loras", [])
    active_lora_adapters = []
    active_lora_weights = []

    if loras_to_load:
        print(f"[generate_image] Attempting to load {len(loras_to_load)} LoRAs.")
        for i, lora_info in enumerate(loras_to_load):
            lora_model_id = lora_info["model_id"]
            lora_weight = lora_info["weight"]
            # Use unique adapter names for each LoRA loaded in this job
            adapter_name = f"job_lora_{i}" 
            try:
                # NOTE: For load_lora_weights with local_files_only=True,
                # the LoRA model (lora_model_id) must be pre-cached.
                # If it needs to be downloaded, set local_files_only=False
                # or ensure your RunPod env pre-downloads and caches it.
                base_pipe.load_lora_weights(
                    lora_model_id,
                    adapter_name=adapter_name,
                    local_files_only=True # Assuming LoRAs are pre-cached
                )
                active_lora_adapters.append(adapter_name)
                active_lora_weights.append(lora_weight)
                MODELS.loaded_job_lora_adapters.append(adapter_name) # Track for cleanup
                print(f"[generate_image] Loaded LoRA: {lora_model_id} with adapter name {adapter_name} and weight {lora_weight}")
            except Exception as e:
                print(f"[generate_image] Failed to load LoRA {lora_model_id}: {e}")
                # Optionally, decide if this should be a fatal error
                # return {"error": f"Failed to load LoRA {lora_model_id}: {e}"}

        if active_lora_adapters:
            base_pipe.set_adapters(active_lora_adapters, adapter_weights=active_lora_weights)
            print(f"[generate_image] Set active LoRA adapters: {active_lora_adapters} with weights: {active_lora_weights}")
    # -------------------------------------------------------------------------

    starting_image = job_input.get("image_url") # Use .get for safer access

    if job_input.get("seed") is None: # Use .get for safer access
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input["seed"])

    base_pipe.scheduler = make_scheduler(
        job_input["scheduler"], base_pipe.scheduler.config
    )

    output_images = None
    try:
        if starting_image:
            init_image = load_image(starting_image).convert("RGB")
            # Note: LoRAs are typically applied to the base model.
            # If you want LoRAs on img2img with refiner, ensure refiner also loads them or logic is adjusted.
            # Current setup applies LoRAs to MODELS.base, which isn't directly used here if only refiner runs.
            # This might mean img2img won't use LoRAs unless MODELS.refiner is also adapted or
            # MODELS.base is used first for an img2img pass before refiner.
            # For simplicity, assuming LoRAs are for txt2img path primarily.
            # If LoRAs should affect this path, MODELS.refiner would need similar LoRA loading logic,
            # or the base_pipe should be used for the initial img2img step.
            print("[generate_image] Img2Img path. LoRAs loaded to base_pipe might not be used unless base_pipe runs.")
            output_images = MODELS.refiner(
                prompt=job_input["prompt"],
                num_inference_steps=job_input["refiner_inference_steps"],
                strength=job_input["strength"], # Ensure strength is passed if it's img2img
                image=init_image,
                generator=generator,
                # If LoRAs are desired here, and refiner is a StableDiffusionXLImg2ImgPipeline,
                # it would also need the LoRAs loaded and set.
            ).images
        else:
            # Text-to-Image generation using base_pipe (with LoRAs if any)
            generated_latents = base_pipe(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                denoising_end=job_input.get("high_noise_frac", 0.8), # Default from example, ensure it's in schema
                output_type="latent",
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
                # cross_attention_kwargs might be used if LoRA weights were not set via set_adapters
                # but set_adapters is the more modern way for multiple LoRAs
            ).images

            # Refine the image
            # Ensure refiner also has LoRAs if they are meant to affect the refining step.
            # Typically, refiner is not affected by base model LoRAs unless specifically designed.
            output_images = MODELS.refiner(
                prompt=job_input["prompt"],
                num_inference_steps=job_input["refiner_inference_steps"],
                strength=job_input.get("strength", 0.3), # Default from example, ensure it's in schema for refiner step
                image=generated_latents,
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            ).images

    except RuntimeError as err:
        # Clean up LoRAs even if there's an error during generation
        MODELS.cleanup_job_loras()
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True,
        }
    except Exception as e: # Catch any other exception
        MODELS.cleanup_job_loras()
        return {
            "error": f"Exception: {e}, Stack Trace: {e.__traceback__}",
            "refresh_worker": True, # May need to refresh worker on unexpected errors
        }

    # Cleanup LoRAs after successful generation
    MODELS.cleanup_job_loras()

    image_urls = _save_and_upload_images(output_images, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0] if image_urls else None,
        "seed": job_input["seed"],
    }

    if starting_image: # This was an img2img job
        results["refresh_worker"] = True # As per original logic

    return results


runpod.serverless.start({"handler": generate_image})
