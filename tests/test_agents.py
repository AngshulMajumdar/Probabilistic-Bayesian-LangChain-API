from fastapi.testclient import TestClient

from bayesian_prob_langchain_api.api.app import create_app

client = TestClient(create_app())


def test_run_local_echo_path():
    response = client.post('/api/v1/agents/run', json={'query': 'say hello', 'backend': 'local'})
    assert response.status_code == 200
    data = response.json()
    assert data['best_action'] in {'ANSWER', 'TOOL'}
    assert isinstance(data['posterior_probs'], list)


def test_run_local_calculator_path():
    response = client.post('/api/v1/agents/run', json={'query': 'calculate 2 + 3', 'backend': 'local'})
    assert response.status_code == 200
    payload = response.json()['best_payload']
    assert payload['name'] == 'calculator'


def test_run_rejects_unknown_backend():
    response = client.post('/api/v1/agents/run', json={'query': 'hello', 'backend': 'unknown'})
    assert response.status_code == 400
    assert 'unsupported_backend' in response.json()['detail']
