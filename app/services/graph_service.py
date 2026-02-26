"""
Graph service: build note-connection graph from indexed links.
"""

from collections import defaultdict
from pathlib import Path

from app.api.models.graph import GraphEdge, GraphNode, GraphResponse
from app.db import db


def _norm_path(value: str) -> str:
    """Normalize path-like strings for matching."""
    return (value or "").replace("\\", "/").strip().lower()


class GraphService:
    """Builds graph payloads for note-link visualization."""

    def get_links_graph(self, max_edges: int = 2000, include_dangling: bool = True) -> GraphResponse:
        all_files = db.get_all_files()

        file_by_id: dict[int, dict] = {int(file_record["id"]): file_record for file_record in all_files}
        file_by_path: dict[str, dict] = {}
        file_by_stem: dict[str, list[dict]] = defaultdict(list)
        for file_record in all_files:
            path = str(file_record["path"])
            normalized = _norm_path(path)
            file_by_path[normalized] = file_record
            stem = Path(path.replace("\\", "/")).stem.lower()
            file_by_stem[stem].append(file_record)

        tags_by_file_id = self._tags_by_file_id()

        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str], GraphEdge] = {}

        with db.cursor() as cur:
            cur.execute(
                """
                SELECT l.from_file_id, l.to_path, f.path AS source_path, f.title AS source_title
                FROM links l
                JOIN files f ON f.id = l.from_file_id
                ORDER BY l.id
                LIMIT ?
                """,
                (max_edges,),
            )
            rows = cur.fetchall()

        for row in rows:
            source_id = int(row["from_file_id"])
            source_file = file_by_id.get(source_id)
            if not source_file:
                continue

            source_path = str(source_file["path"])
            source_node_id = source_path
            if source_node_id not in nodes:
                nodes[source_node_id] = GraphNode(
                    id=source_node_id,
                    label=str(source_file["title"]),
                    path=source_path,
                    node_type="note",
                    tags=tags_by_file_id.get(source_id, []),
                )

            raw_target = str(row["to_path"] or "").strip()
            if not raw_target:
                continue

            target_file = self._resolve_target_file(raw_target, file_by_path, file_by_stem)
            if target_file is not None:
                target_node_id = str(target_file["path"])
                if target_node_id not in nodes:
                    target_file_id = int(target_file["id"])
                    nodes[target_node_id] = GraphNode(
                        id=target_node_id,
                        label=str(target_file["title"]),
                        path=str(target_file["path"]),
                        node_type="note",
                        tags=tags_by_file_id.get(target_file_id, []),
                    )
                edge_key = (source_node_id, target_node_id, raw_target)
                edges[edge_key] = GraphEdge(
                    source=source_node_id,
                    target=target_node_id,
                    label=raw_target,
                    resolved=True,
                )
                continue

            if not include_dangling:
                continue

            target_node_id = f"unresolved::{raw_target}"
            if target_node_id not in nodes:
                nodes[target_node_id] = GraphNode(
                    id=target_node_id,
                    label=raw_target,
                    path=None,
                    node_type="dangling",
                    tags=[],
                )
            edge_key = (source_node_id, target_node_id, raw_target)
            edges[edge_key] = GraphEdge(
                source=source_node_id,
                target=target_node_id,
                label=raw_target,
                resolved=False,
            )

        self._apply_degrees(nodes, list(edges.values()))

        edge_list = sorted(
            edges.values(),
            key=lambda edge: (edge.source, edge.target, edge.label),
        )
        node_list = sorted(
            nodes.values(),
            key=lambda node: (node.node_type, node.id),
        )

        resolved_edges = sum(1 for edge in edge_list if edge.resolved)
        dangling_edges = len(edge_list) - resolved_edges

        return GraphResponse(
            nodes=node_list,
            edges=edge_list,
            total_nodes=len(node_list),
            total_edges=len(edge_list),
            resolved_edges=resolved_edges,
            dangling_edges=dangling_edges,
        )

    @staticmethod
    def _resolve_target_file(
        raw_target: str,
        file_by_path: dict[str, dict],
        file_by_stem: dict[str, list[dict]],
    ) -> dict | None:
        """
        Resolve a parsed outbound link target to an indexed file.

        Matching order:
        1) exact path
        2) exact path + `.md`
        3) unique filename stem
        """
        normalized = _norm_path(raw_target)
        if not normalized:
            return None

        direct = file_by_path.get(normalized)
        if direct is not None:
            return direct

        if "." not in Path(normalized).name:
            md_candidate = file_by_path.get(f"{normalized}.md")
            if md_candidate is not None:
                return md_candidate

        stem = Path(normalized).stem.lower()
        candidates = file_by_stem.get(stem, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _tags_by_file_id() -> dict[int, list[str]]:
        """Collect tags keyed by file_id."""
        by_file: dict[int, list[str]] = defaultdict(list)
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT ft.file_id AS file_id, t.name AS tag_name
                FROM file_tags ft
                JOIN tags t ON t.id = ft.tag_id
                ORDER BY ft.file_id, t.name
                """
            )
            for row in cur.fetchall():
                by_file[int(row["file_id"])].append(str(row["tag_name"]))
        return dict(by_file)

    @staticmethod
    def _apply_degrees(nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
        """Compute in/out/total degree for each node."""
        in_degree: dict[str, int] = defaultdict(int)
        out_degree: dict[str, int] = defaultdict(int)
        for edge in edges:
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1

        for node in nodes.values():
            node.in_degree = in_degree.get(node.id, 0)
            node.out_degree = out_degree.get(node.id, 0)
            node.degree = node.in_degree + node.out_degree


# Singleton
graph_service = GraphService()
