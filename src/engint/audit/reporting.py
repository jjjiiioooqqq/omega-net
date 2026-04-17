from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AuditBundle:
    assumed: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    next_test: list[str] = field(default_factory=list)


class AuditWriter:
    def build_bundle(self, **kwargs) -> AuditBundle:
        return AuditBundle(**kwargs)

    def write_json(self, bundle: AuditBundle, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(bundle)
        # unknown markers are explicit and preserved.
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
