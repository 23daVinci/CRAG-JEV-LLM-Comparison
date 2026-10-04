from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

DEFAULT_PRICING_PATH = Path(__file__).parent.parent.parent.parent / "config" / "pricing.yaml"


@dataclass(frozen=True, slots=True)
class PriceRow:
    provider: str
    model: str
    effective_from: datetime
    effective_to: datetime | None
    input_per_mtok_usd: float
    output_per_mtok_usd: float

    def covers(self, provider: str, model: str, at: datetime) -> bool:
        if self.provider != provider or self.model != model:
            return False
        if at < self.effective_from:
            return False
        return not (self.effective_to is not None and at >= self.effective_to)


class PriceBook:
    def __init__(self, rows: list[PriceRow]) -> None:
        self._rows = rows

    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> PriceBook:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = [
            PriceRow(
                provider=entry["provider"],
                model=entry["model"],
                effective_from=datetime.fromisoformat(entry["effective_from"]).replace(tzinfo=UTC),
                effective_to=(
                    datetime.fromisoformat(entry["effective_to"]).replace(tzinfo=UTC)
                    if entry.get("effective_to")
                    else None
                ),
                input_per_mtok_usd=float(entry["input_per_mtok_usd"]),
                output_per_mtok_usd=float(entry["output_per_mtok_usd"]),
            )
            for entry in raw
        ]
        return cls(rows)

    def cost_usd(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, at: datetime
    ) -> float:
        for row in self._rows:
            if row.covers(provider, model, at):
                return (
                    input_tokens / 1_000_000 * row.input_per_mtok_usd
                    + output_tokens / 1_000_000 * row.output_per_mtok_usd
                )
        raise LookupError(f"no price row covers {provider}/{model} at {at.isoformat()}")
