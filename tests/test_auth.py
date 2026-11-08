"""
Unit & Integration Tests for Authentication and Authorization (Phase 1).
"""
import pytest

def test_register_success(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "SecurePassword123!"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "alice@example.com"
    assert "id" in data
    assert "hashed_password" not in data

def test_register_weak_password(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "weak"}
    )
    assert response.status_code == 422

def test_register_duplicate_email(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "SecurePassword123!"}
    )
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "SecurePassword123!"}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

def test_login_success(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "bob@example.com", "password": "BobPassword999!"}
    )
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": "BobPassword999!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["email"] == "bob@example.com"

def test_login_invalid_password(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": "WrongPassword123!"}
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]

def test_protected_route_unauthenticated(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401

def test_protected_route_authenticated(client):
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": "charlie@example.com", "password": "CharliePassword789!"}
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "charlie@example.com", "password": "CharliePassword789!"}
    )
    token = login_res.json()["access_token"]
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "charlie@example.com"

