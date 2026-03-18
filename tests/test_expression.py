from src.nuro_symbolic_regression.domain.expression import ExpressionNode


def test_expression_eval_linear_form() -> None:
    # (x0 + (2 * x1))
    expr = ExpressionNode(
        "+",
        left=ExpressionNode("x0"),
        right=ExpressionNode("*", left=ExpressionNode(2.0), right=ExpressionNode("x1")),
    )
    value = expr.evaluate({"x0": 1.5, "x1": 4.0})
    assert value == 9.5


def test_expression_replace_subtree() -> None:
    expr = ExpressionNode("+", left=ExpressionNode("x0"), right=ExpressionNode("x1"))
    updated = expr.replace(("R",), ExpressionNode(10.0))
    assert updated.to_infix() == "(x0 + 10.0)"
    assert expr.to_infix() == "(x0 + x1)"

