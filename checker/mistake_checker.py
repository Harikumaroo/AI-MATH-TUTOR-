"""Simple heuristic-based mistake detector for algebraic equations."""
import re
import sympy


def detect_mistakes(latex_str: str, sympy_obj=None) -> list:
    """Detect potential common student mistakes in equation strings or SymPy expressions."""
    mistakes = []
    if not latex_str:
        return mistakes

    # 1. Check for unbalanced parentheses or brackets
    if latex_str.count("(") != latex_str.count(")"):
        mistakes.append("Unbalanced parentheses '(' and ')' detected.")
    if latex_str.count("[") != latex_str.count("]"):
        mistakes.append("Unbalanced brackets '[' and ']' detected.")

    # 2. Check for common distribution error pattern e.g. a(x+b) -> ax + b instead of ax + ab
    distrib_match = re.search(r"(\d+)\s*\(\s*([a-zA-Z])\s*([+-])\s*(\d+)\s*\)\s*=\s*\1\2\s*\3\s*(\d+)", latex_str)
    if distrib_match:
        coeff, var, op, const, result_const = distrib_match.groups()
        if int(result_const) == int(const):
            mistakes.append(f"Distribution error: Forgot to multiply {coeff} by {const} inside parentheses.")

    # 3. Check for division by zero
    if "/0" in latex_str.replace(" ", "") or "/ 0" in latex_str:
        mistakes.append("Division by zero detected.")

    # 4. Check for double operators e.g. ++ or -- or +-
    if re.search(r"[+\-*/]{2,}", latex_str.replace(" ", "")):
        mistakes.append("Consecutive operator symbols (e.g., '++', '+-') detected.")

    return mistakes
