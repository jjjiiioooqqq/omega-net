from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AuditBundle:
    ASSUMED: list[str] = field(default_factory=list)
    DERIVED: list[str] = field(default_factory=list)
    VERIFIED: list[str] = field(default_factory=list)
    UNVERIFIED: list[str] = field(default_factory=list)
    FAILURE_MODES: list[str] = field(default_factory=list)
    NEXT_TEST: list[str] = field(default_factory=list)


class AuditWriter:
    """Emit machine-readable audit reports with strict uncertainty labeling."""

    def build_bundle(self, **kwargs) -> AuditBundle:
        bundle = AuditBundle(**kwargs)
        for section in ("ASSUMED", "DERIVED", "VERIFIED", "UNVERIFIED", "FAILURE_MODES", "NEXT_TEST"):
            values = getattr(bundle, section)
            setattr(bundle, section, [str(v) for v in values])
        return bundle

    def write_json(self, bundle: AuditBundle, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(bundle), fh, indent=2)
