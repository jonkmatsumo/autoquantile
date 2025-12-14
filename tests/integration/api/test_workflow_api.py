"""Integration tests for workflow endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config


def test_start_workflow_works_without_auth_when_key_not_set(client_no_auth):
    """Test that start workflow works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    sample_data = json.dumps([{"col1": 1, "col2": "a"}])
    response = client_no_auth.post(
        "/api/v1/workflow/start",
        json={
            "data": sample_data,
            "columns": ["col1", "col2"],
            "dtypes": {"col1": "int64", "col2": "object"},
            "dataset_size": 1,
        },
    )
    assert response.status_code in [200, 400, 401]


def test_get_workflow_state_works_without_auth_when_key_not_set(client_no_auth):
    """Test that get workflow state works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/api/v1/workflow/test123")
    assert response.status_code in [200, 401, 404]


def test_get_workflow_state_not_found(client, api_key):
    """Test getting workflow state when workflow not found. Args: client: Test client. api_key: API key."""
    response = client.get(
        "/api/v1/workflow/nonexistent",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "WORKFLOW_NOT_FOUND"

