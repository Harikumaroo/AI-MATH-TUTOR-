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
import requests

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

from utils.image_utils import to_bytes
from utils.env_utils import get_api_key

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
                # easyocr returns list of (bbox, text, conf)
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
    # Replace common OCR artefacts
    text = text.replace("×", "*")
    text = text.replace("—", "-")
    text = re.sub(r"[^0-9a-zA-Z\s=+\-*/()\\.^\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_math_from_text(ocr_text: str) -> str:
    """Heuristic: pick the line or substring that looks most like a math expression.

    Strategy:
    - Split into lines, for each line keep only math-allowed characters
    - Score by length and number of math symbols, pick best candidate
    """
    if not ocr_text:
        return ""
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=+-*/^()[]{} \\._%")
    math_lines = []
    for line in ocr_text.splitlines():
        if not line.strip():
            continue
        # remove common leading garbage words
        # keep characters that are typical in math
        filtered = ''.join(ch for ch in line if ch in allowed)
        # remove words like 'Algebra', 'Problems' that may remain
        filtered = re.sub(r"\b(Algebra|Problems|Problem|Exercises)\b", "", filtered, flags=re.IGNORECASE)
        filtered = filtered.strip()
        # compact multiple spaces
        filtered = re.sub(r"\s+", " ", filtered)
        if len(filtered) >= 1:
            math_lines.append(filtered)

    if not math_lines:
        return _clean_ocr_text(ocr_text)

    # score lines: longer and with more math symbols wins
    def score(s):
        math_symbols = sum(1 for ch in s if ch in '=+-*/^()[]{}\\')
        return len(s) + math_symbols * 3

    best = max(math_lines, key=score)
    # final cleanup: replace unicode fraction ½ etc.
    best = best.replace('\u00bd', '1/2')
    # Try to extract the most math-like contiguous substring (e.g. from '0 Algebra 3+18' -> '3+18')
    try:
        # compact spaces and try full string without spaces first
        nospace = re.sub(r"\s+", "", best)
        if re.search(r"[+\-*/^=]", nospace) and re.search(r"\d", nospace):
            return nospace

        # otherwise find candidate contiguous substrings and prefer those with both digits and operators
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


def _call_gemini_api(prompt: str, api_key: str, model: str = "gemini-3.6-flash", image_bytes: bytes = None) -> str:
    """Call Google Gemini REST endpoint with support for vision and text prompts."""
    import base64
    candidate_models = [model, "gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    
    parts = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    parts.append({"text": prompt})
    
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
    }
    
    for m in candidate_models:
        try:
            m_clean = m.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_clean}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=body, timeout=20)
            if resp.status_code == 200:
                j = resp.json()
                if "candidates" in j and j["candidates"]:
                    text = j["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    text = re.sub(r"^```(?:latex)?", "", text.strip(), flags=re.IGNORECASE)
                    text = re.sub(r"```$", "", text.strip()).strip()
                    if text.startswith("$") and text.endswith("$"):
                        text = text[1:-1].strip()
                    return text
        except Exception as e:
            logger.exception("Gemini model %s call failed: %s", m, e)
            continue
            
    return ""


def _call_groq_api(prompt: str, api_key: str, model: str = "openai/gpt-oss-120b") -> str:
    """Call Groq AI API endpoint for fast LaTeX conversion."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512
        }
        resp = requests.post(url, headers=headers, json=body, timeout=15)
        if resp.status_code == 200:
            j = resp.json()
            if "choices" in j and j["choices"]:
                text = j["choices"][0]["message"]["content"].strip()
                text = re.sub(r"^```(?:latex)?", "", text.strip(), flags=re.IGNORECASE)
                text = re.sub(r"```$", "", text.strip()).strip()
                if text.startswith("\\(") and text.endswith("\\)"):
                    text = text[2:-2].strip()
                elif text.startswith("$") and text.endswith("$"):
                    text = text[1:-1].strip()
                return text
    except Exception as e:
        logger.exception("Groq API call failed: %s", e)
    return ""


def llm_convert_to_latex(ocr_text: str, image=None, model="gemini-3.6-flash") -> str:
    """Convert OCR output into clean LaTeX using available LLM providers (Gemini, Groq, OpenAI)."""
    cleaned = _clean_ocr_text(ocr_text)
    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "latex_prompt.txt")

    # Try Google Gemini if provided
    gemini_key = get_api_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            if image:
                prompt = "Extract the math equation or function from this image into a clean LaTeX expression. Return ONLY raw LaTeX without markdown code block syntax."
            elif os.path.exists(prompt_file):
                prompt = open(prompt_file).read().replace("<<OCR_TEXT>>", cleaned)
            else:
                prompt = f"Convert the following math OCR text into a clean LaTeX math expression:\n{cleaned}"
                
            out = _call_gemini_api(prompt, gemini_key, model=model, image_bytes=image)
            if out:
                return out.strip()
        except Exception:
            logger.exception("Gemini conversion attempt failed")

    # Try Groq AI if provided
    groq_key = get_api_key("GROQ_API_KEY")
    if groq_key and cleaned:
        try:
            prompt = f"Convert the following math OCR text into clean LaTeX. Return ONLY the raw LaTeX string without markdown wrapper or explanation:\n{cleaned}"
            out = _call_groq_api(prompt, groq_key)
            if out:
                return out.strip()
        except Exception:
            logger.exception("Groq conversion attempt failed")

    # Next try OpenAI if available
    api_key = get_api_key("OPENAI_API_KEY")
    if api_key and _have_openai and os.path.exists(prompt_file):
        try:
            openai.api_key = api_key
            prompt = open(prompt_file).read().replace("<<OCR_TEXT>>", cleaned)
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            latex = resp["choices"][0]["message"]["content"].strip()
            return latex
        except Exception as e:
            logger.exception("LLM LaTeX conversion (OpenAI) failed: %s", e)

    return cleaned
