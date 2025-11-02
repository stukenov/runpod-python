# my_worker.py

import runpod

def is_even(job):

    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        return {"error": "Silly human, you need to pass an integer."}

    if the_number % 2 == 0:
        return True

    return False

runpod.serverless.start({"handler": is_even})