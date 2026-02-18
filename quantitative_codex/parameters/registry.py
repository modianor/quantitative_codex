from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ParameterVersion:
    strategy: str
    version: str
    params: dict
    note: str
    created_at: str


class ParameterRegistry:
    """Simple JSON-backed parameter version registry."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def _save(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))

    def add(self, strategy: str, version: str, params: dict, note: str = "") -> ParameterVersion:
        rows = self._load()
        item = ParameterVersion(
            strategy=strategy,
            version=version,
            params=params,
            note=note,
            created_at=datetime.utcnow().isoformat(),
        )
        rows.append(asdict(item))
        self._save(rows)
        return item

    def latest(self, strategy: str) -> ParameterVersion | None:
        rows = [r for r in self._load() if r["strategy"] == strategy]
        if not rows:
            return None
        rows = sorted(rows, key=lambda x: x["created_at"])
        last = rows[-1]
        return ParameterVersion(**last)

    def history(self, strategy: str) -> list[ParameterVersion]:
        rows = [r for r in self._load() if r["strategy"] == strategy]
        rows = sorted(rows, key=lambda x: x["created_at"])
        return [ParameterVersion(**r) for r in rows]
