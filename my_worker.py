# filter_service.py
# pip install torch torchvision ftfy regex tqdm pillow
# pip install git+https://github.com/openai/CLIP.git
# (и твой clip_photo_filter.py должен лежать рядом)

import os
import sys
import io
import json
import argparse
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, Dict, Any
from PIL import Image, ImageFile

# доп. устойчивость к "битым" изображениям
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- опционально RunPod ---
try:
    import runpod  # type: ignore
except Exception:
    runpod = None

from clip_photo_filter import ClipPhotoFilter


def _download_or_open(image_ref: str) -> bytes:
    """Принимает http(s) URL или локальный путь. Возвращает bytes изображения."""
    if image_ref.lower().startswith(("http://", "https://")):
        req = urllib.request.Request(image_ref, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
    # локальный путь
    with open(image_ref, "rb") as f:
        return f.read()


def _download_json(url: str) -> Dict[str, Any]:
    """Загружает JSON по URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
        return json.loads(data.decode('utf-8'))


def _analyze_bytes(
    image_bytes: bytes,
    threshold: float = 0.55,
    simple: bool = False,
    unwanted_prompts: Optional[list] = None,
    allowed_prompts: Optional[list] = None
):
    """Общий анализ. simple=True — «попроще» режим для CLI."""
    # размер картинки (для отчёта)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
    except Exception:
        width = height = None

    # В обоих режимах используем один и тот же фильтр, но CLI держим максимально простым
    clf = ClipPhotoFilter(
        unwanted_prompts=unwanted_prompts,
        allowed_prompts=allowed_prompts
    )

    # В «простом» режиме только is_allowed; в расширенном можно добавить score при желании
    is_allowed = clf.is_allowed(image_bytes, threshold=threshold)

    out = {
        "is_allowed": bool(is_allowed),
        "image_width": width,
        "image_height": height,
        "threshold": threshold,
    }
    if not simple:
        # Можно вернуть top-3 классов для отладки на RunPod
        try:
            scores = clf.score(image_bytes)
            top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            out["top3"] = [{"label": k, "prob": float(v)} for k, v in top3]
        except Exception as e:
            out["score_error"] = str(e)
    return out


# ------------------ RunPod handler ------------------

def photo_filter(job):
    """RunPod Serverless handler: ожидает {"input": {"photo_url": "...", "blacklist_url": "..."}}."""
    photo_url = job.get("input", {}).get("photo_url")
    blacklist_url = job.get("input", {}).get("blacklist_url")
    threshold = job.get("input", {}).get("threshold", 0.55)

    if not photo_url:
        return {"output": {"error": "Photo URL is required"}}

    try:
        # Загружаем изображение
        image_bytes = _download_or_open(photo_url)

        # Загружаем blacklist если указан URL
        unwanted_prompts = None
        allowed_prompts = None

        if blacklist_url:
            try:
                blacklist_data = _download_json(blacklist_url)
                unwanted_prompts = blacklist_data.get("unwanted")
                allowed_prompts = blacklist_data.get("allowed")
            except Exception as bl_err:
                # Если не удалось загрузить blacklist, используем дефолтные значения
                print(f"Warning: Failed to load blacklist from {blacklist_url}: {bl_err}")
                print("Using default blacklist values")

        # Анализируем изображение
        result = _analyze_bytes(
            image_bytes,
            threshold=threshold,
            simple=False,
            unwanted_prompts=unwanted_prompts,
            allowed_prompts=allowed_prompts
        )
        result["image_downloaded"] = True
        result["source"] = "runpod"
        result["blacklist_loaded"] = blacklist_url is not None and unwanted_prompts is not None
        return {"output": result}
    except (HTTPError, URLError) as err:
        return {"output": {"error": f"Download failed: {err}"}}
    except Exception as err:
        return {"output": {"error": str(err)}}


# ------------------ CLI (терминал) ------------------

def main_cli(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CLIP photo filter (CLI). Передай URL или путь к файлу."
    )
    parser.add_argument("image", help="HTTP(S) URL или локальный путь к изображению")
    parser.add_argument("--threshold", type=float, default=0.55, help="Порог (0..1)")
    parser.add_argument(
        "--json", action="store_true",
        help="Печатать чистый JSON (без лишнего текста)"
    )
    args = parser.parse_args(argv)

    try:
        image_bytes = _download_or_open(args.image)
        result = _analyze_bytes(image_bytes, threshold=args.threshold, simple=True)
        result["source"] = "cli"
        result["image_downloaded"] = args.image.startswith(("http://", "https://"))

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (HTTPError, URLError) as err:
        err_obj = {"error": f"Download failed: {err}", "source": "cli"}
        print(json.dumps(err_obj, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except FileNotFoundError as err:
        err_obj = {"error": f"File not found: {err}", "source": "cli"}
        print(json.dumps(err_obj, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    except Exception as err:
        err_obj = {"error": str(err), "source": "cli"}
        print(json.dumps(err_obj, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


# ------------------ Entry point ------------------

def _should_run_runpod() -> bool:
    """Эвристика: если установлен модуль runpod и есть признаки окружения RunPod."""
    if runpod is None:
        return False
    # Часто встречающиеся переменные окружения на RunPod/Serverless
    env_markers = [
        "RUNPOD_POD_ID",
        "RUNPOD_TASK_ID",
        "RUNPOD_ENDPOINT_ID",
        "RUNPOD_AI_APP",
    ]
    return any(os.environ.get(k) for k in env_markers) or hasattr(runpod, "serverless")


if __name__ == "__main__":
    if _should_run_runpod():
        # Режим RunPod Serverless
        runpod.serverless.start({"handler": photo_filter})
    else:
        # Обычный терминал / локальный запуск
        raise SystemExit(main_cli())


