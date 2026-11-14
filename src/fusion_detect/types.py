"""Shared result type: every detector (provenance, watermark, passive) returns this."""
from dataclasses import dataclass, field
from typing import Literal, Optional

Layer = Literal["provenance", "watermark", "passive", "metadata"]


@dataclass
class DetectorResult:
    detector: str                  # e.g. "ojha_clip", "npr", "c2pa", "stablesig"
    layer: Layer
    score: Optional[float]         # raw P(AI) or detector score; None = abstain (signal absent)
    hard: Optional[bool] = None    # for gate layers: True=AI proven, False=real proven, None=no verdict
    evidence: dict = field(default_factory=dict)   # e.g. {"bit_acc": 0.97} or C2PA assertion summary

    @property
    def abstained(self) -> bool:
        return self.score is None and self.hard is None
