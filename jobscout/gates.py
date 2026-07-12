"""Config models for the gated + tiered assessment.

Kept separate from jobscout/config.py (which is provider/model wiring) because these
describe the gate engine's rule surface. Structured gates reference the canonical
feature vocabulary in features.py; the priority tiers rank the apply set.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GateConfig(BaseModel):
    """One hard eligibility gate: fail => decision = no (non-compensatory).

    Either a structured predicate over a canonical feature (feature + op + value)
    or a natural-language rule the LLM answers pass/fail/unknown (rule=...).
    """

    name: str
    # Structured form — feature must be in the canonical vocabulary (features.py).
    feature: str = ""
    op: str = ""  # one of GATE_OPS
    value: Any = None
    # NL escape hatch — mutually exclusive with the structured form.
    rule: str = ""
    # What to do when the fact is missing or below min_confidence:
    # pass = give the benefit of the doubt; fail = reject; pass_flag = pass but
    # flag the gate as unknown (feeds the priority uncertainty penalty).
    on_unknown: Literal["pass", "fail", "pass_flag"] = "pass"
    # A fail verdict below this confidence is treated as unknown, so a hard fail
    # only ever comes from a high-confidence detection.
    min_confidence: float = 0.7

    @model_validator(mode="after")
    def check_shape(self) -> GateConfig:
        from .features import FEATURES, GATE_OPS

        structured = bool(self.feature or self.op or self.value is not None)
        if structured == bool(self.rule):
            raise ValueError(
                f"gate {self.name!r}: set either feature/op/value or rule, not both/neither"
            )
        if structured:
            if self.feature not in FEATURES:
                raise ValueError(
                    f"gate {self.name!r}: unknown feature {self.feature!r} — use one of "
                    f"{sorted(FEATURES)} or a natural-language rule"
                )
            if self.op not in GATE_OPS:
                raise ValueError(f"gate {self.name!r}: op must be one of {GATE_OPS}")
        return self


class TierConfig(BaseModel):
    """One priority tier: a listing lands here by title match or role substance."""

    name: str
    titles: list[str] = Field(default_factory=list)
    # role_substance values (see features.py) that also place a role in this tier —
    # substance beats title, so a Staff-titled founding role can land in a
    # founding tier. Best (earliest-listed) tier wins.
    substances: list[str] = Field(default_factory=list)


class PriorityConfig(BaseModel):
    tiers: list[TierConfig] = Field(default_factory=list)
    # Subtracted from the within-tier rank (and the derived fit) per unknown gate.
    unknown_penalty: int = 2
