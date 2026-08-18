"""SymPy parsing, equation solving, and step-by-step resolution generation."""
import re
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

transformations = standard_transformations + (implicit_multiplication_application,)


def clean_latex_string(latex_str: str) -> str:
    """Convert LaTeX notation into SymPy parseable Python math syntax."""
    if not latex_str:
        return ""
    s = latex_str.strip()
    s = s.replace("$", "").replace("\\", "")
    s = s.replace("×", "*").replace("cdot", "*").replace("times", "*")
    s = s.replace("÷", "/")
    
    # Handle \frac{a}{b} pattern
    s = re.sub(r"frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"((\1)/(\2))", s)
    
    # Replace exponents x^2 -> x**2
    s = s.replace("^", "**")
    
    # Remove remaining LaTeX commands
    s = re.sub(r"[a-zA-Z]+\{([^}]+)\}", r"(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    
    return s.strip()


def parse_latex_to_sympy(latex_str: str):
    """Parse a LaTeX string or raw math expression into a SymPy object (Eq or Expr)."""
    cleaned = clean_latex_string(latex_str)
    if not cleaned:
        raise ValueError("Empty math expression")

    # Check for equality sign '='
    if "=" in cleaned:
        parts = cleaned.split("=")
        if len(parts) == 2:
            lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
            lhs = parse_expr(lhs_str, transformations=transformations) if lhs_str else sympy.Integer(0)
            rhs = parse_expr(rhs_str, transformations=transformations) if rhs_str else sympy.Integer(0)
            return sympy.Eq(lhs, rhs)
    
    # Otherwise parse as single expression
    return parse_expr(cleaned, transformations=transformations)


def solve_equation(sympy_obj):
    """Solve SymPy Eq or Expr for free symbols. Returns dict with 'solutions'."""
    if isinstance(sympy_obj, sympy.Eq):
        symbols = sorted(list(sympy_obj.free_symbols), key=lambda s: s.name)
        target_sym = symbols[0] if symbols else sympy.Symbol("x")
        sols = sympy.solve(sympy_obj, target_sym)
        return {"solutions": sols, "symbol": target_sym}
    else:
        # Solve f(x) = 0
        symbols = sorted(list(sympy_obj.free_symbols), key=lambda s: s.name)
        target_sym = symbols[0] if symbols else sympy.Symbol("x")
        sols = sympy.solve(sympy_obj, target_sym)
        return {"solutions": sols, "symbol": target_sym}


def generate_steps(sympy_obj):
    """Generate step-by-step solution cards for the given equation/expression."""
    steps = []
    
    if isinstance(sympy_obj, sympy.Eq):
        lhs, rhs = sympy_obj.lhs, sympy_obj.rhs
        symbols = sorted(list(sympy_obj.free_symbols), key=lambda s: s.name)
        target_sym = symbols[0] if symbols else sympy.Symbol("x")
        
        steps.append(("Identify Equation", f"{sympy.latex(lhs)} = {sympy.latex(rhs)}"))
        
        # Step: Move rhs to lhs -> lhs - rhs = 0
        diff = sympy.simplify(lhs - rhs)
        steps.append(("Standard Form (LHS - RHS = 0)", f"{sympy.latex(diff)} = 0"))
        
        # Step: Expand / Simplify
        expanded = sympy.expand(diff)
        if expanded != diff:
            steps.append(("Expand Terms", f"{sympy.latex(expanded)} = 0"))
            
        # Step: Solve
        sols = sympy.solve(sympy_obj, target_sym)
        if sols:
            sol_str = ", ".join([sympy.latex(s) for s in sols])
            steps.append(("Solve for Variable", f"{target_sym.name} = {sol_str}"))
        else:
            steps.append(("Solve for Variable", "No analytical solutions found."))
    else:
        symbols = sorted(list(sympy_obj.free_symbols), key=lambda s: s.name)
        target_sym = symbols[0] if symbols else sympy.Symbol("x")
        steps.append(("Given Expression", sympy.latex(sympy_obj)))
        
        simplified = sympy.simplify(sympy_obj)
        steps.append(("Simplified Expression", sympy.latex(simplified)))
        
        sols = sympy.solve(sympy_obj, target_sym)
        if sols:
            sol_str = ", ".join([sympy.latex(s) for s in sols])
            steps.append(("Roots / Solutions (Expr = 0)", f"{target_sym.name} = {sol_str}"))

    return steps
