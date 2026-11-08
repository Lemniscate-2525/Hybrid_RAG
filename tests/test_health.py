"""
Tests for Observability and Health Probes (Phase 8 & 10).
"""
import pytest

def test_liveness_probe(client):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "version" in data
    assert "timestamp" in data

def test_readiness_probe(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["UP", "DOWN"]
    assert "components" in data
    assert "database" in data["components"]
    assert "redis" in data["components"]
    assert "qdrant" in data["components"]

