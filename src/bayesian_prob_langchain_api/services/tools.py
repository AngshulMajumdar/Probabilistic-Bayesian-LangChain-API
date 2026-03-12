from __future__ import annotations

from typing import Any, Dict, List
import ast
import operator as op


class EchoTool:
    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"echo": payload}


class CalculatorTool:
    ALLOWED = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self.ALLOWED:
            return self.ALLOWED[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.ALLOWED:
            return self.ALLOWED[type(node.op)](self._eval(node.operand))
        raise ValueError("unsupported_expression")

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        expr = str(payload.get("expression", ""))
        tree = ast.parse(expr, mode="eval")
        return {"expression": expr, "result": self._eval(tree.body)}


class RetrieverTool:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query", "")).lower().strip()
        qset = set(query.split())
        scored = []
        for doc in self.docs:
            words = set(str(doc["text"]).lower().split())
            score = len(qset & words)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [{"id": d["id"], "text": d["text"], "score": score} for score, d in scored]
        return {"query": query, "hits": hits}
