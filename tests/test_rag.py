from fastapi.testclient import TestClient

from bayesian_prob_langchain_api.api.app import create_app

client = TestClient(create_app())


def test_rag_query_returns_hits():
    response = client.post('/api/v1/rag/query', json={'query': 'backend agnostic runtime', 'top_k': 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data['hits']) >= 1
    assert 'backend agnostic' in data['answer'].lower()


def test_rag_query_handles_no_hits():
    response = client.post('/api/v1/rag/query', json={'query': 'unmatched_term_xyz', 'top_k': 2})
    assert response.status_code == 200
    data = response.json()
    assert data['hits'] == []
    assert data['answer'] == 'No relevant documents found.'
