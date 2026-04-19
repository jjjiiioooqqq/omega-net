"""Explicit local graph schema for engineering provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass(slots=True)
class NodeRecord:
    """Typed node representation in the digital thread graph."""

    node_id: str
    node_type: str
    payload: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Graph-backed store for assumptions, equations, outputs, and evidence."""

    ALLOWED_NODE_TYPES = {
        "equation",
        "assumption",
        "derived_constraint",
        "geometry_version",
        "solver_output",
        "experiment",
        "material_data",
        "claim",
        "unknown",
        "citation",
        "design_conclusion",
    }

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, record: NodeRecord) -> None:
        if record.node_type not in self.ALLOWED_NODE_TYPES:
            raise ValueError(f"Unsupported node type: {record.node_type}")
        self.graph.add_node(record.node_id, node_type=record.node_type, payload=record.payload)

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        if src not in self.graph or dst not in self.graph:
            raise KeyError("Both source and destination nodes must exist before linking")
        self.graph.add_edge(src, dst, relation=relation)

    def has_provenance(self, claim_id: str) -> bool:
        """Check whether claim traces to citation + assumptions + equation/solver outputs."""
        if claim_id not in self.graph:
            return False
        predecessors = nx.ancestors(self.graph, claim_id)
        types = {self.graph.nodes[n]["node_type"] for n in predecessors}
        return "citation" in types and "assumption" in types and (
            "solver_output" in types or "equation" in types
        )

    def export_json(self) -> dict[str, Any]:
        """Machine-friendly export for persistence or audit inclusion."""
        nodes = [
            {"id": n, "type": d["node_type"], "payload": d["payload"]}
            for n, d in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "relation": d["relation"]}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
