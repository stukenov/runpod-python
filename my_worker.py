# my_worker.py

import runpod
import numpy as np
import torch
from pkg_resources import packaging
import clip

torch_version = torch.__version__

clip_models = ", ".join(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

models_info = (
    f"Input resolution: {input_resolution}\n"
    f"Context length: {context_length}\n"
    f"Vocab size: {vocab_size}"
)

def is_even(job):
    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        return {"error": "Silly human, you need to pass an integer."}

    if the_number % 2 == 0:
        result = f"Even {torch_version} {clip_models} {models_info}"
    else:
        result = f"Odd {torch_version} {clip_models} {models_info}"
    return {"result": result}

runpod.serverless.start({"handler": is_even})