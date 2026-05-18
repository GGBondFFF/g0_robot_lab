from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def literal_assignment(module: ast.Module, name: str) -> Any:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment not found: {name}")


def numeric_assignment_expr(module: ast.Module, name: str) -> ast.expr:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"Numeric assignment not found: {name}")


def eval_numeric_expr(expr: ast.expr, names: dict[str, float]) -> float:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return float(expr.value)
    if isinstance(expr, ast.Name):
        return float(names[expr.id])
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        key = f"{expr.value.id}.{expr.attr}"
        return float(names[key])
    if isinstance(expr, ast.BinOp):
        left = eval_numeric_expr(expr.left, names)
        right = eval_numeric_expr(expr.right, names)
        if isinstance(expr.op, ast.Add):
            return left + right
        if isinstance(expr.op, ast.Sub):
            return left - right
        if isinstance(expr.op, ast.Mult):
            return left * right
        if isinstance(expr.op, ast.Div):
            return left / right
        if isinstance(expr.op, ast.Pow):
            return left**right
    raise AssertionError(f"Unsupported numeric expression: {ast.dump(expr)}")


def class_def(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class not found: {name}")


def nested_class_def(parent: ast.ClassDef, name: str) -> ast.ClassDef:
    for node in parent.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Nested class not found: {parent.name}.{name}")


def keyword_value_from_call(class_node: ast.ClassDef, assignment_name: str, keyword: str) -> ast.expr:
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == assignment_name:
                    call = node.value
                    if not isinstance(call, ast.Call):
                        raise AssertionError(f"{assignment_name} is not assigned from a call")
                    for item in call.keywords:
                        if item.arg == keyword:
                            return item.value
    raise AssertionError(f"Keyword not found: {assignment_name}.{keyword}")


def literal_keyword_value(class_node: ast.ClassDef, assignment_name: str, keyword: str) -> Any:
    return ast.literal_eval(keyword_value_from_call(class_node, assignment_name, keyword))
