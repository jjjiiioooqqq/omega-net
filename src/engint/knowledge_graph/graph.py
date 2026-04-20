from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx


class NodeType(str, Enum):
    EQUATION = "equation"
    ASSUMPTION = "assumption"
    DERIVED_CONSTRAINT = "derived_constraint"
    GEOMETRY = "geometry"
    SOLVER_OUTPUT = "solver_output"
    EXPERIMENT = "experiment"
    MATERIAL = "material"
    CLAIM = "claim"


@dataclass(slots=True)
class NodeRecord:
    node_id: str
    node_type: NodeType
    payload: dict[str, object] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    critical: bool = False


class DigitalThreadGraph:
    """Graph-backed digital thread with explicit schema and provenance checks."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, record: NodeRecord) -> None:
        self.graph.add_node(
            record.node_id,
            node_type=record.node_type.value,
            payload=record.payload,
            provenance=record.provenance,
            critical=record.critical,
        )

    def relate(self, source: str, relation: str, target: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def claims_missing_provenance(self) -> list[str]:
        missing: list[str] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.CLAIM.value and not data.get("provenance"):
                missing.append(str(node_id))
        return missing

    def critical_nodes_missing_provenance(self) -> list[str]:
        missing: list[str] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("critical") and not data.get("provenance"):
                missing.append(str(node_id))
        return missing

    def trace(self, node_id: str) -> dict[str, object]:
        preds = list(self.graph.predecessors(node_id))
        succs = list(self.graph.successors(node_id))
        data = dict(self.graph.nodes[node_id])
        return {"node": node_id, "data": data, "parents": preds, "children": succs}
