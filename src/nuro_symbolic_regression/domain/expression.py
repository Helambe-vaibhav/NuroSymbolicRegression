from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .operators import DEFAULT_OPERATOR_SET, OP_ARITY, OP_FUNC


@dataclass
class ExpressionNode:
    value: str | float
    left: "ExpressionNode | None" = None
    right: "ExpressionNode | None" = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def clone(self) -> "ExpressionNode":
        return ExpressionNode(
            value=self.value,
            left=self.left.clone() if self.left else None,
            right=self.right.clone() if self.right else None,
        )

    def to_infix(self) -> str:
        if self.is_leaf():
            return str(self.value)
        if self.right is None:
            return f"{self.value}({self.left.to_infix()})"
        return f"({self.left.to_infix()} {self.value} {self.right.to_infix()})"

    def size(self) -> int:
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def depth(self) -> int:
        return 0 if self.is_leaf() else 1 + max(self.left.depth() if self.left else 0, self.right.depth() if self.right else 0)

    def evaluate(
        self,
        variables: dict[str, float],
        operators: dict[str, Callable[..., float]] | None = None,
        clip_value: float = 1e10,
    ) -> float:
        func_map: dict[str, Callable[..., float]] = OP_FUNC if operators is None else operators
        arity_map = OP_ARITY if operators is None else {name: DEFAULT_OPERATOR_SET[name].arity for name in operators}
        try:
            if isinstance(self.value, (int, float)):
                result = float(self.value)
            elif self.value in variables:
                result = float(variables[self.value])
            elif self.value in func_map:
                arity = arity_map[self.value]
                op_func = func_map[self.value]
                if arity == 1 and self.left is not None:
                    result = op_func(self.left.evaluate(variables, func_map, clip_value))
                elif arity == 2 and self.left is not None and self.right is not None:
                    left_val = self.left.evaluate(variables, func_map, clip_value)
                    right_val = self.right.evaluate(variables, func_map, clip_value)
                    result = op_func(left_val, right_val)
                else:
                    return 1.0
            else:
                result = float(self.value)
        except Exception:
            return 1.0
        return max(min(result, clip_value), -clip_value)

    def paths(self) -> list[tuple[str, ...]]:
        """Return all node paths in pre-order using 'L'/'R' edges."""
        items: list[tuple[str, ...]] = [tuple()]
        if self.left is not None:
            items.extend(("L",) + p for p in self.left.paths())
        if self.right is not None:
            items.extend(("R",) + p for p in self.right.paths())
        return items

    def node_at(self, path: Iterable[str]) -> "ExpressionNode":
        node = self
        for step in path:
            if step == "L":
                if node.left is None:
                    raise ValueError("Invalid path: missing left child")
                node = node.left
            elif step == "R":
                if node.right is None:
                    raise ValueError("Invalid path: missing right child")
                node = node.right
            else:
                raise ValueError(f"Invalid step: {step}")
        return node

    def replace(self, path: tuple[str, ...], subtree: "ExpressionNode") -> "ExpressionNode":
        if len(path) == 0:
            return subtree.clone()

        root = self.clone()
        cursor = root
        for step in path[:-1]:
            cursor = cursor.left if step == "L" else cursor.right
            if cursor is None:
                raise ValueError("Invalid path for replacement")

        if path[-1] == "L":
            cursor.left = subtree.clone()
        elif path[-1] == "R":
            cursor.right = subtree.clone()
        else:
            raise ValueError(f"Invalid edge marker: {path[-1]}")
        return root

