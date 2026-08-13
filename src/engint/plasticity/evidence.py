from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """An immutable ingested artifact; the original is never silently changed."""

    document_id: str
    sha256: str
    source: str
    doc_type: str
    retrieved_at: datetime
    version: int


@dataclass(slots=True)
class DocumentStore:
    """Content-addressed evidence store.

    If a webpage or file changes tomorrow, D17 != D17': registering the new
    content creates a new version alongside the old one instead of mutating
    it. Claims reference exact document versions, so provenance survives
    upstream edits.
    """

    documents: dict[str, Document] = field(default_factory=dict)
    _by_source: dict[str, list[str]] = field(default_factory=dict)

    def register(
        self,
        content: bytes,
        source: str,
        doc_type: str = "document",
        now: datetime | None = None,
    ) -> Document:
        digest = hashlib.sha256(content).hexdigest()
        versions = self._by_source.setdefault(source, [])
        for doc_id in versions:
            if self.documents[doc_id].sha256 == digest:
                return self.documents[doc_id]  # identical content: no new version
        document = Document(
            document_id=f"D-{digest[:12]}",
            sha256=digest,
            source=source,
            doc_type=doc_type,
            retrieved_at=now or datetime.now(timezone.utc),
            version=len(versions) + 1,
        )
        self.documents[document.document_id] = document
        versions.append(document.document_id)
        return document

    def register_file(self, path: Path, doc_type: str = "document") -> Document:
        return self.register(path.read_bytes(), source=str(path.name), doc_type=doc_type)

    def versions_of(self, source: str) -> list[Document]:
        return [self.documents[doc_id] for doc_id in self._by_source.get(source, [])]

    def verify(self, document_id: str, content: bytes) -> bool:
        return self.documents[document_id].sha256 == hashlib.sha256(content).hexdigest()
