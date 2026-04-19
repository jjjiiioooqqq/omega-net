from engineering_intelligence.knowledge_graph import KnowledgeGraph, NodeRecord


def test_claim_provenance_traceability() -> None:
    graph = KnowledgeGraph()
    graph.add_node(NodeRecord("a", "assumption", {}))
    graph.add_node(NodeRecord("e", "equation", {}))
    graph.add_node(NodeRecord("s", "solver_output", {}))
    graph.add_node(NodeRecord("c", "citation", {}))
    graph.add_node(NodeRecord("claim", "claim", {}))
    for src in ("a", "e", "s", "c"):
        graph.add_edge(src, "claim", "supports")

    assert graph.has_provenance("claim")
