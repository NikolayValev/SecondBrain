"""
Graph-related API models for note connection visualization.
"""

from typing import Optional

from pydantic import BaseModel


class GraphNode(BaseModel):
    """A node in the note-link graph."""
    id: str
    label: str
    path: Optional[str] = None
    node_type: str  # note | dangling
    tags: list[str] = []
    in_degree: int = 0
    out_degree: int = 0
    degree: int = 0


class GraphEdge(BaseModel):
    """A directed edge in the note-link graph."""
    source: str
    target: str
    label: str
    resolved: bool = True


class GraphResponse(BaseModel):
    """Graph payload for frontend visualization."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int
    total_edges: int
    resolved_edges: int
    dangling_edges: int
