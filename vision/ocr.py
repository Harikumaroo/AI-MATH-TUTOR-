"""OCR and LLM conversion helpers.

This module provides a fallback pipeline:
- OCR via EasyOCR (preferred) or pytesseract
- LLM call wrapper for converting OCR output to clean LaTeX (optional)
"""
import os
import logging
from PIL import Image
import re
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
    _have_easyocr = True
except Exception:
    _have_easyocr = False

try:
    import pytesseract
    _have_tesseract = True
except Exception:
    _have_tesseract = False

try:
    import openai
    _have_openai = True
except Exception:
    _have_openai = False
    import requests

from utils.image_utils import to_bytes

class OCREngine:
    def __init__(self, lang_list=["en"]):
        self.lang_list = lang_list
        if _have_easyocr:
            try:
                self.reader = easyocr.Reader(lang_list, gpu=False)
            except Exception:
                self.reader = None
        else:
            self.reader = None

    def extract_text(self, pil_image):
        """Extract raw text from the image using available OCR engines."""
        try:
            if self.reader:
                result = self.reader.readtext(np.asarray(pil_image))
                texts = [r[1] for r in result]
                raw = "\n".join(texts)
                return raw

            if _have_tesseract:
                txt = pytesseract.image_to_string(pil_image)
                return txt

            return ""
        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return ""

    def extract_math(self, pil_image):
        """Return a cleaned math expression derived from OCR output."""
        raw = self.extract_text(pil_image)
        try:
            math = _extract_math_from_text(raw)
            if math:
                return math
        except Exception:
            logger.exception("Math extraction failed; falling back to raw OCR")
        return raw


def _clean_ocr_text(ocr_text: str) -> str:
    """Basic cleaning heuristics from OCR to LaTeX-like string."""
    if not ocr_text:
        return ""
    text = ocr_text.strip()
    text = text.replace("×", "*")
    text = text.replace("—", "-")
    text = re.sub(r"[^0-9a-zA-Z\s=+\-*/()\\.^\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_math_from_text(ocr_text: str) -> str:
    """Heuristic: pick the line or substring that looks most like a math expression."""
    if not ocr_text:
        return ""
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=+-*/^()[]{} \\._%")
    math_lines = []
    for line in ocr_text.splitlines():
        if not line.strip():
            continue
        filtered = ''.join(ch for ch in line if ch in allowed)
        filtered = re.sub(r"\b(Algebra|Problems|Problem|Exercises)\b", "", filtered, flags=re.IGNORECASE)
        filtered = filtered.strip()
        filtered = re.sub(r"\s+", " ", filtered)
        if len(filtered) >= 1:
            math_lines.append(filtered)

    if not math_lines:
        return _clean_ocr_text(ocr_text)

    def score(s):
        math_symbols = sum(1 for ch in s if ch in '=+-*/^()[]{}\\')
        return len(s) + math_symbols * 3

    best = max(math_lines, key=score)
    best = best.replace('\u00bd', '1/2')
    try:
        nospace = re.sub(r"\s+", "", best)
        if re.search(r"[+\-*/^=]", nospace) and re.search(r"\d", nospace):
            return nospace

        candidates = re.findall(r"[0-9A-Za-z\)\]\{\}\\\^\/\*+\-\(\)]+", best)
        def is_math_like(c):
            return bool(re.search(r"[+\-*/^=]", c) and re.search(r"\d", c))

        math_cands = [c for c in candidates if is_math_like(c)]
        if math_cands:
            math_best = max(math_cands, key=lambda x: (len(x), sum(1 for ch in x if ch in '+-*/^=')))
            return math_best
    except Exception:
        pass

    return best


def _call_gemini_api(prompt: str, api_key: str, model: str = "models/text-bison-001") -> str:
    """Call Google Generative Models REST endpoint (Gemini/PaLM) with an API key."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta2/{model}:generate?key={api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "prompt": {"text": prompt},
            "temperature": 0.0,
            "maxOutputTokens": 512
        }
        resp = requests.post(url, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        j = resp.json()
        out = ""
        if isinstance(j, dict):
            if "candidates" in j and len(j["candidates"]) > 0:
                out = j["candidates"][0].get("output", "")
            elif "output" in j:
                out = j.get("output", "")
        return out
    except Exception as e:
        logger.exception("Gemini API call failed: %s", e)
        return ""


def llm_convert_to_latex(ocr_text: str, image=None, model="gpt-4o-mini") -> str:
    """Convert OCR output into clean LaTeX using available LLM providers."""
    cleaned = _clean_ocr_text(ocr_text)
    prompt_file = os.path.join(os.path.dirname(__file__), "..", "prompts", "latex_prompt.txt")

    # Try Google Gemini / PaLM if provided
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and os.path.exists(prompt_file):
        try:
            prompt = open(prompt_file).read().replace("<<OCR_TEXT>>", cleaned)
            out = _call_gemini_api(prompt, gemini_key, model="models/text-bison-001")
            if out:
                return out.strip()
        except Exception:
            logger.exception("Gemini conversion attempt failed")

    # Next try OpenAI if available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and _have_openai and os.path.exists(prompt_file):
        try:
            openai.api_key = api_key
            prompt = open(prompt_file).read().replace("<<OCR_TEXT>>", cleaned)
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            latex = resp["choices"][0]["message"]["content"].strip()
            return latex
        except Exception as e:
            logger.exception("LLM LaTeX conversion (OpenAI) failed: %s", e)

    return cleaned
