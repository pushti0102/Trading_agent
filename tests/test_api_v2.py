import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_v2_agents_registry(client):
    """Verify v2 agents endpoint returns all 15 strategies across categories."""
    response = client.get("/api/v2/agents")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    agents = data["agents"]
    assert len(agents) == 15

    agent_ids = {a["id"] for a in agents}
    assert "ppo_trader" in agent_ids
    assert "a2c_trader" in agent_ids
    assert "ddpg_trader" in agent_ids
    assert "double_dueling_dqn" in agent_ids
    assert "validation_weighted_ensemble" in agent_ids
    assert "risk_parity" in agent_ids


def test_v2_walk_forward_folds_and_summary(client):
    """Verify walk-forward endpoints return valid expanding folds and aggregate metrics."""
    res_folds = client.get("/api/v2/walk-forward/folds")
    assert res_folds.status_code == 200
    folds_data = res_folds.json()
    assert "records" in folds_data
    assert len(folds_data["records"]) > 0

    res_sum = client.get("/api/v2/walk-forward/summary")
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert "summary" in sum_data
    assert len(sum_data["summary"]) > 0


def test_v2_robustness_data(client):
    """Verify robustness endpoint returns cost sensitivity and bootstrap estimates."""
    response = client.get("/api/v2/robustness")
    assert response.status_code == 200
    data = response.json()
    assert "cost_sensitivity" in data
    assert "regime_stress_tests" in data
    assert "bootstrap_confidence_intervals" in data
    assert "ablation_reward_and_risk" in data
    assert len(data["cost_sensitivity"]) > 0
    assert len(data["bootstrap_confidence_intervals"]) > 0


def test_v2_reports_manifest(client):
    """Verify master versioned reports manifest reflects completed research phases."""
    response = client.get("/api/v2/reports/manifest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "OliveTrade 2.0"
    assert "phases" in data
    phase_ids = [p["phase_id"] for p in data["phases"]]
    assert "phase_01_audit" in phase_ids
    assert "phase_07_robustness" in phase_ids


def test_v2_csv_exports(client):
    """Verify research CSV exports return valid CSV MIME content."""
    res_cost = client.get("/api/v2/export/robustness.csv")
    assert res_cost.status_code == 200
    assert "text/csv" in res_cost.headers["content-type"]
    assert "quoted_spread_pct" in res_cost.text

    res_wf = client.get("/api/v2/export/walk-forward.csv")
    assert res_wf.status_code == 200
    assert "text/csv" in res_wf.headers["content-type"]
    assert "strategy" in res_wf.text
