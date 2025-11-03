# my_worker.py
import runpod
from clip_photo_filter import ClipPhotoFilter
import urllib.request
from urllib.error import URLError, HTTPError


def photo_filter(job):
    photo_url = job["input"].get("photo_url")
    if not photo_url:
        return {"output": {"error": "Photo URL is required"}}

    try:
        # Download the image from the internet and pass bytes to the filter
        request = urllib.request.Request(photo_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            image_bytes = response.read()

        clf = ClipPhotoFilter()
        is_allowed = clf.is_allowed(image_bytes, threshold=0.55)
        return {"output": {"is_allowed": is_allowed}}
    except (HTTPError, URLError) as e:
        return {"output": {"error": f"Download failed: {e}"}}
    except Exception as e:
        return {"output": {"error": str(e)}}

runpod.serverless.start({"handler": photo_filter})
