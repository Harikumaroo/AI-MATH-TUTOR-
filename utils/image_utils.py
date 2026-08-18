"""Image processing utilities for AI Math Tutor."""
import io
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_image(pil_image):
    """Convert PIL image or file path into OpenCV BGR image array."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image)
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(pil_image, max_dim=1600):
    """Resize image, convert to grayscale, and enhance contrast/sharpness for OCR."""
    try:
        w, h = pil_image.size
        scale = min(1.0, float(max_dim) / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        gray = pil_image.convert("L")
        enhanced = ImageOps.autocontrast(gray)
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
        )
        return enhanced
    except Exception:
        return pil_image.convert("L")


def to_bytes(pil_image, fmt="PNG"):
    """Convert PIL Image into byte string."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()
