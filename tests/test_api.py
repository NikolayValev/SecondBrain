"""
End-to-end tests for the FastAPI application.
Tests all API endpoints with a real test client.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import os

from fastapi.testclient import TestClient


# We need to set up the environment before importing the app
@pytest.fixture(scope="module")
def test_vault():
    """Create a test vault for E2E tests."""
    vault_path = Path(tempfile.mkdtemp())
    
    # Create sample files
    (vault_path / "note1.md").write_text("""---
title: First Note
tags:
  - python
  - testing
---

# First Note

This is the first note about Python programming.

## Section One

Content in section one with [[Second Note]] link.

## Section Two

More content with #inline-tag.
""", encoding="utf-8")
    
    (vault_path / "note2.md").write_text("""# Second Note

A note about testing.

#testing #automation

Links to [[First Note]] and [[Third Note]].
""", encoding="utf-8")
    
    (vault_path / "note3.md").write_text("""# Third Note

Some searchable content here.

Unique term: xyzzy123

#unique-tag
""", encoding="utf-8")
    
    # Create subdirectory with file
    subdir = vault_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("""# Nested Note

A note in a subdirectory.

#nested
""", encoding="utf-8")
    
    yield vault_path
    
    # Cleanup
    shutil.rmtree(vault_path, ignore_errors=True)


@pytest.fixture(scope="module")
def test_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture(scope="module")
def client(test_vault, test_db_path):
    """Create a test client with patched configuration."""
    # Patch the config before importing the app
    with patch.dict(os.environ, {
        "VAULT_PATH": str(test_vault),
        "DATABASE_PATH": test_db_path,
        "LOG_LEVEL": "WARNING"
    }):
        # Need to reload config to pick up new env vars
        import importlib
        import app.config
        importlib.reload(app.config)
        
        # Now import and create the app
        from app.main import app
        
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health check should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should have expected structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "vault_path" in data
        assert "watcher_running" in data
    
    def test_health_status_healthy(self, client):
        """Health status should be healthy."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
    
    def test_health_watcher_running(self, client):
        """Watcher should be running."""
        response = client.get("/health")
        data = response.json()
        
        assert data["watcher_running"] is True


class TestStatsEndpoint:
    """Tests for GET /stats endpoint."""
    
    def test_stats_returns_200(self, client):
        """Stats should return 200."""
        response = client.get("/stats")
        assert response.status_code == 200
    
    def test_stats_response_structure(self, client):
        """Stats response should have expected fields."""
        response = client.get("/stats")
        data = response.json()
        
        assert "file_count" in data
        assert "section_count" in data
        assert "tag_count" in data
        assert "link_count" in data
        assert "last_indexed" in data
    
    def test_stats_file_count(self, client):
        """Should have indexed the test files."""
        response = client.get("/stats")
        data = response.json()
        
        assert data["file_count"] == 4  # 4 files in test vault
    
    def test_stats_has_sections(self, client):
        """Should have indexed sections."""
        response = client.get("/stats")
        data = response.json()
        
        assert data["section_count"] > 0
    
    def test_stats_has_tags(self, client):
        """Should have indexed tags."""
        response = client.get("/stats")
        data = response.json()
        
        assert data["tag_count"] > 0
    
    def test_stats_has_links(self, client):
        """Should have indexed links."""
        response = client.get("/stats")
        data = response.json()
        
        assert data["link_count"] > 0
    
    def test_stats_has_last_indexed(self, client):
        """Should have last indexed timestamp."""
        response = client.get("/stats")
        data = response.json()
        
        assert data["last_indexed"] is not None


class TestSearchEndpoint:
    """Tests for GET /search endpoint."""
    
    def test_search_returns_200(self, client):
        """Search should return 200 for valid query."""
        response = client.get("/search?q=Python")
        assert response.status_code == 200
    
    def test_search_response_structure(self, client):
        """Search response should have expected structure."""
        response = client.get("/search?q=Python")
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert "count" in data
    
    def test_search_finds_content(self, client):
        """Search should find matching content."""
        response = client.get("/search?q=Python")
        data = response.json()
        
        assert data["count"] > 0
        assert len(data["results"]) > 0
    
    def test_search_result_structure(self, client):
        """Search results should have expected fields."""
        response = client.get("/search?q=Python")
        data = response.json()
        
        if data["count"] > 0:
            result = data["results"][0]
            assert "file_path" in result
            assert "title" in result
            assert "heading" in result
            assert "snippet" in result
            assert "rank" in result
    
    def test_search_returns_query(self, client):
        """Response should echo the query."""
        response = client.get("/search?q=testing")
        data = response.json()
        
        assert data["query"] == "testing"
    
    def test_search_unique_term(self, client):
        """Search should find unique terms."""
        response = client.get("/search?q=xyzzy123")
        data = response.json()
        
        assert data["count"] >= 1
        assert any("note3.md" in r["file_path"] for r in data["results"])
    
    def test_search_limit_parameter(self, client):
        """Search should respect limit parameter."""
        response = client.get("/search?q=note&limit=2")
        data = response.json()
        
        assert len(data["results"]) <= 2
    
    def test_search_empty_query_rejected(self, client):
        """Empty query should be rejected."""
        response = client.get("/search?q=")
        assert response.status_code == 422  # Validation error
    
    def test_search_no_results(self, client):
        """Search with no matches should return empty results."""
        response = client.get("/search?q=nonexistentterm987654321")
        data = response.json()
        
        assert data["count"] == 0
        assert data["results"] == []
    
    def test_search_missing_query_rejected(self, client):
        """Missing query parameter should be rejected."""
        response = client.get("/search")
        assert response.status_code == 422


class TestFileEndpoint:
    """Tests for GET /file endpoint."""
    
    def test_file_returns_200(self, client):
        """Should return 200 for existing file."""
        response = client.get("/file?path=note1.md")
        assert response.status_code == 200
    
    def test_file_response_structure(self, client):
        """File response should have expected structure."""
        response = client.get("/file?path=note1.md")
        data = response.json()
        
        assert "path" in data
        assert "title" in data
        assert "content" in data
    
    def test_file_returns_content(self, client):
        """Should return file content."""
        response = client.get("/file?path=note1.md")
        data = response.json()
        
        assert "First Note" in data["content"]
        assert "Python programming" in data["content"]
    
    def test_file_returns_title(self, client):
        """Should return correct title."""
        response = client.get("/file?path=note1.md")
        data = response.json()
        
        assert data["title"] == "First Note"
    
    def test_file_not_found(self, client):
        """Should return 404 for non-existent file."""
        response = client.get("/file?path=nonexistent.md")
        assert response.status_code == 404
    
    def test_file_nested_path(self, client):
        """Should handle nested file paths."""
        # Try both path separators
        response = client.get("/file?path=subdir/nested.md")
        if response.status_code == 404:
            response = client.get("/file?path=subdir\\nested.md")
        
        assert response.status_code == 200
        data = response.json()
        assert "Nested Note" in data["content"]
    
    def test_file_missing_path_rejected(self, client):
        """Missing path parameter should be rejected."""
        response = client.get("/file")
        assert response.status_code == 422


class TestTagsEndpoint:
    """Tests for GET /tags endpoint."""
    
    def test_tags_returns_200(self, client):
        """Tags should return 200."""
        response = client.get("/tags")
        assert response.status_code == 200
    
    def test_tags_response_structure(self, client):
        """Tags response should have expected structure."""
        response = client.get("/tags")
        data = response.json()
        
        assert "tags" in data
        assert "count" in data
    
    def test_tags_list_not_empty(self, client):
        """Should have tags from test files."""
        response = client.get("/tags")
        data = response.json()
        
        assert data["count"] > 0
        assert len(data["tags"]) > 0
    
    def test_tags_have_names_and_counts(self, client):
        """Each tag should have name and file count."""
        response = client.get("/tags")
        data = response.json()
        
        if data["count"] > 0:
            tag = data["tags"][0]
            assert "name" in tag
            assert "file_count" in tag
    
    def test_tags_include_expected(self, client):
        """Should include tags from test files."""
        response = client.get("/tags")
        data = response.json()
        
        tag_names = [t["name"] for t in data["tags"]]
        assert "python" in tag_names or "testing" in tag_names


class TestBacklinksEndpoint:
    """Tests for GET /backlinks endpoint."""
    
    def test_backlinks_returns_200(self, client):
        """Backlinks should return 200."""
        response = client.get("/backlinks?path=First Note")
        assert response.status_code == 200
    
    def test_backlinks_response_structure(self, client):
        """Backlinks response should have expected structure."""
        response = client.get("/backlinks?path=First Note")
        data = response.json()
        
        assert "target" in data
        assert "backlinks" in data
        assert "count" in data
    
    def test_backlinks_finds_links(self, client):
        """Should find files that link to target."""
        # Second Note links to First Note
        response = client.get("/backlinks?path=First Note")
        data = response.json()
        
        assert data["count"] >= 1
        backlink_paths = [b["path"] for b in data["backlinks"]]
        assert any("note2" in p for p in backlink_paths)
    
    def test_backlinks_no_links(self, client):
        """Should handle files with no backlinks."""
        response = client.get("/backlinks?path=nonexistent")
        data = response.json()
        
        assert data["count"] == 0
        assert data["backlinks"] == []
    
    def test_backlinks_missing_path_rejected(self, client):
        """Missing path parameter should be rejected."""
        response = client.get("/backlinks")
        assert response.status_code == 422


class TestReindexEndpoint:
    """Tests for POST /reindex endpoint."""
    
    def test_reindex_returns_200(self, client):
        """Reindex should return 200."""
        response = client.post("/reindex")
        assert response.status_code == 200
    
    def test_reindex_response_structure(self, client):
        """Reindex response should have expected structure."""
        response = client.post("/reindex")
        data = response.json()
        
        assert "status" in data
        assert "indexed" in data
        assert "errors" in data
        assert "type" in data
    
    def test_reindex_incremental_default(self, client):
        """Default reindex should be incremental."""
        response = client.post("/reindex")
        data = response.json()
        
        assert data["type"] == "incremental"
        assert data["status"] == "completed"
    
    def test_reindex_full(self, client):
        """Full reindex should be triggered with parameter."""
        response = client.post("/reindex?full=true")
        data = response.json()
        
        assert data["type"] == "full"
        assert data["status"] == "completed"
    
    def test_reindex_no_errors(self, client):
        """Reindex should complete without errors."""
        response = client.post("/reindex")
        data = response.json()
        
        assert data["errors"] == 0


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""
    
    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_swagger_ui_available(self, client):
        """Swagger UI should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self, client):
        """ReDoc should be accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_for_unknown_endpoint(self, client):
        """Unknown endpoints should return 404."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Wrong HTTP method should return 405."""
        response = client.post("/health")
        assert response.status_code == 405
    
    def test_search_invalid_limit(self, client):
        """Invalid limit should return validation error."""
        response = client.get("/search?q=test&limit=-1")
        assert response.status_code == 422
    
    def test_search_limit_too_high(self, client):
        """Limit above maximum should return validation error."""
        response = client.get("/search?q=test&limit=1000")
        assert response.status_code == 422
