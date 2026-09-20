import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.config import settings

HEADERS = {"x-api-key": settings.api_key}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_search_requires_api_key(client):
    response = client.get("/search", params={"q": "backpropagation"})
    assert response.status_code in (401, 422)


def test_search_returns_results(client):
    response = client.get(
        "/search",
        params={"q": "backpropagation", "top_k": 5, "mode": "hybrid"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "backpropagation"
    assert data["mode"] == "hybrid"
    assert len(data["results"]) == 5


def test_search_respects_top_k(client):
    response = client.get(
        "/search",
        params={"q": "neural network", "top_k": 3, "mode": "bm25"},
        headers=HEADERS,
    )
    data = response.json()
    assert len(data["results"]) == 3


def test_search_invalid_mode_rejected(client):
    response = client.get(
        "/search",
        params={"q": "test", "mode": "not_a_real_mode"},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_search_result_has_required_fields(client):
    response = client.get(
        "/search",
        params={"q": "reinforcement learning", "top_k": 1, "mode": "hybrid"},
        headers=HEADERS,
    )
    result = response.json()["results"][0]
    assert "doc_id" in result
    assert "title" in result
    assert "snippet" in result
    assert "url" in result
    assert "tags" in result


def test_search_tag_filter(client):
    response = client.get(
        "/search",
        params={"q": "neural network", "top_k": 5, "tag": "reinforcement-learning"},
        headers=HEADERS,
    )
    data = response.json()
    assert response.status_code == 200
    for r in data["results"]:
        assert "reinforcement-learning" in [t.lower() for t in r["tags"]]