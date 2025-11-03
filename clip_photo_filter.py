# Зависимости:
# pip install torch torchvision ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
from typing import Union, Optional, Sequence, Dict, Any
import io
import numpy as np
from PIL import Image
import torch
import clip


class ClipPhotoFilter:
    """
    Фильтр изображений на базе CLIP.
    Решает: исключить ли изображение как скриншот/слайд/чек/ценник и т.п.

    Интерфейс:
        - is_allowed(image, threshold=0.50) -> bool
        - score(image) -> dict с вероятностями по классам

    Параметры конструктора:
        unwanted_prompts: список подсказок (классов), которые нужно исключать
        allowed_prompts:  список «нормальных» подсказок для нормализации softmax
        model_name:       "ViT-B/32" по умолчанию
        device:           'cuda' | 'cpu' (по умолчанию авто)
    """

    DEFAULT_UNWANTED = [
        # Скриншоты / экран
        "a smartphone screenshot",
        "a photo of a phone screen",
        "a computer screen capture",
        "a screen recording still",
        # Презентации
        "a presentation slide",
        "a PowerPoint slide",
        "a keynote slide",
        # Чеки / инвойсы / документы оплаты
        "a paper receipt",
        "a shopping receipt",
        "an invoice document",
        "a bill or check",
        # Ценники / ярлыки
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

    def __init__(
        self,
        unwanted_prompts: Optional[Sequence[str]] = None,
        allowed_prompts: Optional[Sequence[str]] = None,
        *,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
    ) -> None:
        self.unwanted_prompts = list(unwanted_prompts or self.DEFAULT_UNWANTED)
        self.allowed_prompts = list(allowed_prompts or self.DEFAULT_ALLOWED)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy init полей модели/препроцессора/текстовых эмбеддингов
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._texts: Sequence[str] = []

    # ---------- Публичный API ----------

    def is_allowed(
        self,
        image: Union[str, bytes, Image.Image, np.ndarray],
        *,
        threshold: float = 0.50,
    ) -> bool:
        """
        Возвращает:
            False — если картинка отнесена к нежелательной категории (исключить),
            True  — если разрешена.
        """
        probs = self._probs(image)
        # максимум среди нежелательных классов
        n_unw = len(self.unwanted_prompts)
        max_unwanted = float(probs[:n_unw].max().item())
        return max_unwanted < threshold

    def score(
        self,
        image: Union[str, bytes, Image.Image, np.ndarray],
    ) -> Dict[str, float]:
        """
        Возвращает вероятности softmax по всем промптам (unwanted + allowed).
        Ключи — текстовые классы.
        """
        probs = self._probs(image)
        return {t: float(p) for t, p in zip(self._texts, probs.tolist())}

    # ---------- Внутреннее ----------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model, preprocess = clip.load(self.model_name, device=self.device, jit=False)
        model.eval()
        self._model = model
        self._preprocess = preprocess

        # Подготавливаем текстовые эмбеддинги
        self._texts = self.unwanted_prompts + self.allowed_prompts
        with torch.no_grad():
            text_tokens = clip.tokenize(self._texts).to(self.device)
            text_features = self._model.encode_text(text_tokens)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
                    img = Image.fromarray(image[:, :, :3]).convert("RGB")
            else:
                raise ValueError("Unsupported numpy array shape for image.")
        else:
            raise TypeError("image must be a path, bytes, PIL.Image, or numpy.ndarray")
        return img

    @torch.inference_mode()
    def _probs(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> torch.Tensor:
        self._ensure_model()
        img = self._ensure_pil(image)
        image_input = self._preprocess(img).unsqueeze(0).to(self.device)
        image_features = self._model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self._text_features.T
        probs = logits.softmax(dim=-1).squeeze(0)
        return probs


# -------- Пример использования --------
if __name__ == "__main__":
    clf = ClipPhotoFilter()  # можно передать device="cpu" или свой список классов
    # path = "example.jpg"
    # print("allowed:", clf.is_allowed(path, threshold=0.55))
    # print("full scores:", clf.score(path))
