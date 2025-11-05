# pip install torch torchvision ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
from typing import Union, Optional, Sequence, Dict, List
import io
import numpy as np
from PIL import Image, ImageFile
import torch
import clip
import warnings
import math
import os
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # детерминированность ближе к "ноутбучной"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClipPhotoFilter:
    """
    Фильтр изображений на базе CLIP (ноутбук-лайк версия).
    - Prompt ensembling: несколько шаблонов на класс
    - Использует model.logit_scale (важно!)
    - Float32 для стабильности
    - Чанкинг токенов текстов
    """

    # Классы (как у тебя)
    DEFAULT_UNWANTED = [
        "a smartphone screenshot",
        "a photo of a phone screen",
        "a computer screen capture",
        "a screen recording still",
        "a presentation slide",
        "a PowerPoint slide",
        "a keynote slide",
        "a paper receipt",
        "a shopping receipt",
        "an invoice document",
        "a bill or check",
        "a price tag",
        "a shelf label with price",
        "a barcode label with price",
    ]

    DEFAULT_ALLOWED = [
        "a natural photo",
        "a candid photo",
        "a portrait photo",
        "a landscape photo",
        "a travel photo",
        "a family photo",
        "a pet photo",
    ]

    # Набор промпт-шаблонов (как делают в ноутбуках/репозиториях с zero-shot)
    PROMPT_TEMPLATES = [
        "a photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
        "a bright photo of {}.",
        "a dark photo of {}.",
        "a low resolution photo of {}.",
        "a high resolution photo of {}.",
        "an image of {}.",
        "a blurry photo of {}.",
        "a good photo of {}.",
        "a clean photo of {}.",
        "a jpeg photo of {}.",
        # Для документов/скринов полезно:
        "a screenshot of {}.",
        "a document photo of {}.",
        "a scanned image of {}.",
    ]

    def __init__(
        self,
        unwanted_prompts: Optional[Sequence[str]] = None,
        allowed_prompts: Optional[Sequence[str]] = None,
        *,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        seed: int = 42,
        text_batch_size: int = 64,   # для длинных списков промптов
        use_float32: bool = True,    # в ноутбуках это часто повышает стабильность
    ) -> None:
        _set_seed(seed)
        self.unwanted_prompts = list(unwanted_prompts or self.DEFAULT_UNWANTED)
        self.allowed_prompts = list(allowed_prompts or self.DEFAULT_ALLOWED)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_batch_size = int(text_batch_size)
        self.use_float32 = use_float32

        self._model = None
        self._preprocess = None
        self._class_names: List[str] = []      # список «чистых» имён классов
        self._text_features = None             # тензор [num_classes, d]
        self._logit_scale = None               # scalar

    # ---------- Публичный API ----------

    @torch.inference_mode()
    def is_allowed(
        self,
        image: Union[str, bytes, Image.Image, np.ndarray],
        *,
        threshold: float = 0.50,
    ) -> bool:
        """
        False — если попало в нежелательные (max вероятн. среди unwanted >= threshold)
        True  — иначе.
        """
        probs = self._probs(image)  # [num_classes]
        n_unw = len(self.unwanted_prompts)
        max_unwanted = float(probs[:n_unw].max().item())
        return max_unwanted < threshold

    @torch.inference_mode()
    def score(
        self,
        image: Union[str, bytes, Image.Image, np.ndarray],
    ) -> Dict[str, float]:
        """
        Возвращает вероятности softmax по классам (unwanted + allowed)
        Ключи — названия классов (как в твоём списке).
        """
        probs = self._probs(image)
        # map к исходным строкам (unwanted + allowed)
        full_names = self.unwanted_prompts + self.allowed_prompts
        return {name: float(p) for name, p in zip(full_names, probs.tolist())}

    # ---------- Внутреннее ----------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        # Загрузка модели и препроцессора как в ноутбуках
        model, preprocess = clip.load(self.model_name, device=self.device, jit=False)
        model.eval()

        # Тип: ближе к ноутбучному сценарию — держим float32 (часто точнее)
        if self.device.startswith("cuda") and not self.use_float32:
            model = model.half()

        self._model = model
        self._preprocess = preprocess

        # logit_scale — ключевой множитель для «жаркости» логитов
        with torch.no_grad():
            self._logit_scale = self._model.logit_scale.exp().detach()

        # Собираем «имена классов» (как у тебя: отдельные строки, не категориальные метки)
        # Мы НЕ схлопываем близкие строки — берём их как отдельные классы,
        # чтобы соответствовать твоему интерфейсу 1:1.
        self._class_names = (self.unwanted_prompts + self.allowed_prompts)

        # Считаем эмбеддинги с prompt-ensembling и усреднением
        self._text_features = self._build_text_features(self._class_names)  # [C, d]

    @torch.inference_mode()
    def _build_text_features(self, class_names: Sequence[str]) -> torch.Tensor:
        """
        Для каждого класса делаем несколько текстовых шаблонов и усредняем эмбеддинги.
        Итог: тензор [num_classes, d], L2-нормированный построчно.
        """
        dtype = torch.float32 if (self.use_float32 or self.device == "cpu") else torch.float16
        all_features = []

        for cls in class_names:
            texts = [templ.format(cls) for templ in self.PROMPT_TEMPLATES]
            # Токенизируем по чанкам (стабильно для длинных списков)
            feats_per_template = []
            for chunk_start in range(0, len(texts), self.text_batch_size):
                chunk = texts[chunk_start:chunk_start + self.text_batch_size]
                text_tokens = clip.tokenize(chunk).to(self.device)
                if dtype == torch.float16:
                    self._model = self._model.half()
                # encode_text -> [n_chunk, d]
                enc = self._model.encode_text(text_tokens).to(torch.float32)  # приводим к fp32 для усреднения
                enc = enc / enc.norm(dim=-1, keepdim=True)
                feats_per_template.append(enc)
            # [T, d] -> mean -> [d]
            cls_feat = torch.cat(feats_per_template, dim=0).mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm()
            all_features.append(cls_feat)

        text_features = torch.stack(all_features, dim=0)  # [C, d]
        return text_features

    def _ensure_pil(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    img = Image.fromarray(image.squeeze(-1)).convert("RGB")
                else:
                    img = Image.fromarray(image[:, :, :3])
                    if img.mode != "RGB":
                        img = img.convert("RGB")
            else:
                raise ValueError("Unsupported numpy array shape for image.")
        else:
            raise TypeError("image must be a path, bytes, PIL.Image, or numpy.ndarray")
        return img

    @torch.inference_mode()
    def _probs(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> torch.Tensor:
        self._ensure_model()
        img = self._ensure_pil(image)

        # Препроцессор из clip.load (как в ноутбуке)
        image_input = self._preprocess(img).unsqueeze(0).to(self.device)

        # Типы: энкодим в fp32 для стабильности логитов
        image_features = self._model.encode_image(image_input).to(torch.float32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # !!! КЛЮЧЕВОЕ: используем logit_scale (как в reference ноутбуках)
        logits = (image_features @ self._text_features.T) * self._logit_scale  # [1, C]
        probs = logits.softmax(dim=-1).squeeze(0)  # [C]
        return probs


# --- Пример ---
if __name__ == "__main__":
    clf = ClipPhotoFilter()
    # path = "example.jpg"
    # print("allowed:", clf.is_allowed(path, threshold=0.55))
    # print("full scores:", clf.score(path))
