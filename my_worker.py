# my_worker.py
import runpod
import torch
import clip

# Инициализируем вне хендлера (рекомендация RunPod)
torch_version = torch.__version__
clip_models = list(clip.available_models())

model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
model.eval()

info = {
    "input_resolution": getattr(model.visual, "input_resolution", None),
    "context_length": getattr(model, "context_length", None),
    "vocab_size": getattr(model, "vocab_size", None),
}

def is_even(job):
    n = job["input"].get("number")
    if not isinstance(n, int):
        return {"error": "Silly human, you need to pass an integer."}

    parity = "even" if n % 2 == 0 else "odd"

    # Возвращаем СТРУКТУРИРОВАННЫЙ ответ + под ключом 'output' для совместимости
    return {
        "output": {
            "parity": parity,
            "torch_version": torch_version,
            "clip_models": clip_models,         # список, не длинная строка
            "model_info": info                  # без \n, чистый JSON
        }
    }

runpod.serverless.start({"handler": is_even})
