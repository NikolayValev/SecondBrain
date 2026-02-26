"""
Tests for graph visualization endpoint.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import register_middleware
from app.api.routes.graph import router as graph_router
from app.config import config
import app.services.graph_service as graph_module


def _build_app(monkeypatch) -> FastAPI:
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["testserver"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["http://localhost:3000"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)

    app = FastAPI()
    register_middleware(app)
    app.include_router(graph_router)
    return app


def _seed_graph_data(temp_db):
    a_id = temp_db.upsert_file("notes/a.md", 100.0, "Note A", "A content")
    b_id = temp_db.upsert_file("notes/b.md", 101.0, "Note B", "B content")
    c_id = temp_db.upsert_file("journal/c.md", 102.0, "Note C", "C content")

    project_tag_id = temp_db.get_or_create_tag("project")
    temp_db.add_file_tag(a_id, project_tag_id)

    temp_db.add_link(a_id, "b")  # stem-resolved -> notes/b.md
    temp_db.add_link(b_id, "missing-note")
    temp_db.add_link(c_id, "notes/a.md")


def test_graph_links_endpoint_returns_nodes_and_edges(monkeypatch, temp_db):
    _seed_graph_data(temp_db)
    monkeypatch.setattr(graph_module, "db", temp_db)

    app = _build_app(monkeypatch)
    client = TestClient(app)
    headers = {"X-API-Key": "x" * 32}

    response = client.get("/graph/links?max_edges=100&include_dangling=true", headers=headers)
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_edges"] == 3
    assert payload["resolved_edges"] == 2
    assert payload["dangling_edges"] == 1

    node_ids = {node["id"] for node in payload["nodes"]}
    assert "notes/a.md" in node_ids
    assert "notes/b.md" in node_ids
    assert "journal/c.md" in node_ids
    assert "unresolved::missing-note" in node_ids

    a_node = next(node for node in payload["nodes"] if node["id"] == "notes/a.md")
    assert "project" in a_node["tags"]
    assert a_node["in_degree"] == 1
    assert a_node["out_degree"] == 1

    resolved_edge = next(
        edge
        for edge in payload["edges"]
        if edge["source"] == "notes/a.md" and edge["target"] == "notes/b.md"
    )
    assert resolved_edge["resolved"] is True


def test_graph_links_endpoint_can_exclude_dangling(monkeypatch, temp_db):
    _seed_graph_data(temp_db)
    monkeypatch.setattr(graph_module, "db", temp_db)

    app = _build_app(monkeypatch)
    client = TestClient(app)
    headers = {"X-API-Key": "x" * 32}

    response = client.get("/graph/links?include_dangling=false", headers=headers)
    assert response.status_code == 200

    payload = response.json()
    assert payload["dangling_edges"] == 0
    assert payload["resolved_edges"] == 2
    assert all(node["node_type"] != "dangling" for node in payload["nodes"])


def test_graph_links_endpoint_requires_auth(monkeypatch, temp_db):
    _seed_graph_data(temp_db)
    monkeypatch.setattr(graph_module, "db", temp_db)

    app = _build_app(monkeypatch)
    client = TestClient(app)

    response = client.get("/graph/links")
    assert response.status_code == 401
