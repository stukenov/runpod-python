# my_worker.py

import runpod

import numpy as np
import torch
from pkg_resources import packaging

torch_version = torch.__version__
import clip

clip_models = clip.available_models()

def is_even(job):

    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        return {"error": "Silly human, you need to pass an integer."}

    result = ""
    if the_number % 2 == 0:
        result = "Even " + torch_version
        result += " " + clip_models[0]
    else:
        result = "Odd " + torch_version
        result += " " + clip_models[1]

    return {"result": result}

runpod.serverless.start({"handler": is_even})