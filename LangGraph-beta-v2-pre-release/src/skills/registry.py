from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ActionRisk(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    HIGH_RISK = "high_risk"


@dataclass
class Skill:
    name: str
    fn: Callable
    spec: dict                     # JSON schema the model sees
    risk: ActionRisk
    data_class_max: str            # highest task classification permitted
    roles: list[str]               # which nodes may call it
    idempotent: bool = True

SKILLS: dict[str, Skill] = {}

def skill(*, risk, data_class_max, roles, idempotent=True):
    def deco(fn):
        import inspect
        from typing import get_type_hints
        hints, sig = get_type_hints(fn), inspect.signature(fn)
        jt = {str: "string", int: "integer", bool: "boolean", float: "number"}
        props = {n: {"type": jt.get(hints.get(n, str), "string")}
                 for n in sig.parameters}
        required = [n for n, p in sig.parameters.items()
                    if p.default is inspect.Parameter.empty]
        SKILLS[fn.__name__] = Skill(
            name=fn.__name__, fn=fn,
            spec={"type": "function", "function": {
                "name": fn.__name__,
                "description": inspect.getdoc(fn) or "",
                "parameters": {"type": "object", "properties": props,
                               "required": required}}},
            risk=risk, data_class_max=data_class_max,
            roles=roles, idempotent=idempotent)
        return fn
    return deco


def specs_for(role: str, data_class: str) -> list[dict]:
    """Only skills this role may call at this classification."""
    rank = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    return [s.spec for s in SKILLS.values()
            if role in s.roles and rank[data_class] <= rank[s.data_class_max]]
