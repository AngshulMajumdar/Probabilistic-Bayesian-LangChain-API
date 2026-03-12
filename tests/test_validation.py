from fastapi.testclient import TestClient

from bayesian_prob_langchain_api.api.app import create_app

client = TestClient(create_app())


def test_empty_query_rejected_for_run():
    response = client.post('/api/v1/agents/run', json={'query': '', 'backend': 'local'})
    assert response.status_code == 422


def test_empty_query_rejected_for_rag():
    response = client.post('/api/v1/rag/query', json={'query': '', 'top_k': 2})
    assert response.status_code == 422


def test_invalid_top_k_rejected():
    response = client.post('/api/v1/rag/query', json={'query': 'test', 'top_k': 0})
    assert response.status_code == 422
