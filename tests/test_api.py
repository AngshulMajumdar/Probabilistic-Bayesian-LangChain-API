from fastapi.testclient import TestClient

from bayesian_prob_langchain_api.api.app import create_app

client = TestClient(create_app())


def test_health():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = response.json()
    assert data['ok'] is True
    assert 'local' in data['available_backends']


def test_info():
    response = client.get('/api/v1/info')
    assert response.status_code == 200
    assert response.json()['api_prefix'] == '/api/v1'


def test_list_tools():
    response = client.get('/api/v1/tools')
    assert response.status_code == 200
    assert set(response.json()['tools']) == {'calculator', 'echo', 'retriever'}
