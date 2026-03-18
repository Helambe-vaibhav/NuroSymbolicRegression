from __future__ import annotations

import sympy as sp

from ...domain.expression import ExpressionNode


def to_sympy(node: ExpressionNode) -> sp.Expr:
    if isinstance(node.value, (int, float)):
        return sp.Float(node.value)

    if node.left is None and node.right is None:
        return sp.Symbol(str(node.value))

    if node.value == "+":
        return to_sympy(node.left) + to_sympy(node.right)
    if node.value == "-":
        return to_sympy(node.left) - to_sympy(node.right)
    if node.value == "*":
        return to_sympy(node.left) * to_sympy(node.right)
    if node.value == "/":
        return to_sympy(node.left) / to_sympy(node.right)
    if node.value == "^":
        return to_sympy(node.left) ** to_sympy(node.right)
    if node.value == "sin":
        return sp.sin(to_sympy(node.left))
    if node.value == "cos":
        return sp.cos(to_sympy(node.left))
    if node.value == "exp":
        return sp.exp(to_sympy(node.left))
    if node.value == "log":
        return sp.log(to_sympy(node.left))

    raise ValueError(f"Unsupported node value: {node.value}")


def to_latex(node: ExpressionNode) -> str:
    return sp.latex(to_sympy(node))

