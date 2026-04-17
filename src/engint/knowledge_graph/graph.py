from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass(slots=True)
class NodeRecord:
    node_id: str
    node_type: str
    payload: dict[str, object] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)


class DigitalThreadGraph:
    """Explicit schema for equations, assumptions, constraints, geometry, and evidence."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, record: NodeRecord) -> None:
        self.graph.add_node(
            record.node_id,
            node_type=record.node_type,
            payload=record.payload,
            provenance=record.provenance,
        )

    def relate(self, source: str, relation: str, target: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def claims_missing_provenance(self) -> list[str]:
        missing: list[str] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "claim" and not data.get("provenance"):
                missing.append(str(node_id))
        return missing

    def trace(self, node_id: str) -> dict[str, object]:
        preds = list(self.graph.predecessors(node_id))
        succs = list(self.graph.successors(node_id))
        data = dict(self.graph.nodes[node_id])
        return {"node": node_id, "data": data, "parents": preds, "children": succs}
