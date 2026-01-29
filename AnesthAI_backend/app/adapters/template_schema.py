'''What: Build a small utility that (a) generates a JSON Schema from your template, and (b) validates model output.
Why: LLMs will sometimes output invalid JSON or wrong keys; this gives you guardrails.'''

from __future__ import annotations
from typing import Any, Dict
import copy

def template_to_json_schema(template: Any) -> Dict[str, Any]:
    """
    Infer a JSON Schema from a template where:
      "" => string
      0  => number
      false => boolean
      [] => array (items inferred from first element if present)
      {} => object
      null => accept any type (no constraints)
    Also enforces 'additionalProperties': False for objects (strict keys).
    """
    def infer(node: Any) -> Dict[str, Any]:
        if node is None:
            return {}  # allow anything
        if isinstance(node, bool):
            return {"type": "boolean"}
        if isinstance(node, int) and not isinstance(node, bool):
            return {"type": "integer"}
        if isinstance(node, float):
            return {"type": "number"}
        if isinstance(node, str):
            return {"type": "string"}
        if isinstance(node, list):
            schema = {"type": "array"}
            if len(node) >= 1:
                schema["items"] = infer(node[0])
            else:
                schema["items"] = {}
            return schema
        if isinstance(node, dict):
            props = {}
            required = []
            for k, v in node.items():
                props[k] = infer(v)
                required.append(k)
            return {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False
            }
        # fallback
        return {}

    base = infer(copy.deepcopy(template))
    # Wrap to ensure it's a full schema document
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **base
    }
    return schema
