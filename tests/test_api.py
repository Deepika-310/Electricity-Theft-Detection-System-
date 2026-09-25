import backend.main as main


def test_fraud_cases_returns_only_anomalies(client):
    """Regression: /fraud-cases used to filter on anomaly == 1, i.e. the NORMAL rows."""
    cases = client.get("/fraud-cases").json()
    assert cases, "expected at least one flagged reading"
    assert all(row["anomaly"] == -1 for row in cases)

    scores = [row["anomaly_score"] for row in cases]
    assert scores == sorted(scores, reverse=True)


def test_fraud_cases_are_a_subset_of_detect(client):
    detect = client.get("/detect").json()
    cases = client.get("/fraud-cases").json()
    assert {r["id"] for r in cases} == {r["id"] for r in detect if r["anomaly"] == -1}


def test_detect_filters_by_meter_and_limit(client):
    rows = client.get("/detect", params={"meter_id": "M01", "limit": 5}).json()
    assert len(rows) == 5
    assert {r["meter_id"] for r in rows} == {"M01"}


def test_flagged_readings_are_enriched_with_actual_theft(client):
    cases = client.get("/fraud-cases").json()
    precision = sum(r["is_theft"] for r in cases) / len(cases)
    assert precision > 0.3  # far above the ~5% base rate


def test_predict_without_meter_uses_neutral_context(client):
    body = client.post("/predict", json={"consumption": 15.0, "voltage": 220.0}).json()
    assert body["context"] == "no_history"
    assert body["is_fraud"] == (body["prediction"] == -1)


def test_stalled_meter_scores_higher_than_a_normal_reading_on_every_meter(client, detector):
    flagged = 0
    for meter in ("M00", "M01", "M02", "M03", "M04"):
        history = detector._load(meter)
        same_hour_yesterday = float(history["consumption"].iloc[-23])  # the next reading is last + 1h
        normal = client.post(
            "/predict", json={"consumption": same_hour_yesterday, "voltage": 230, "meter_id": meter}
        ).json()
        stalled = client.post("/predict", json={"consumption": 0.0, "voltage": 230, "meter_id": meter}).json()

        assert stalled["context"] == "history"
        assert stalled["anomaly_score"] > normal["anomaly_score"]
        assert normal["is_fraud"] is False
        flagged += stalled["is_fraud"]
    assert flagged >= 4


def test_predict_validation_and_unknown_meter(client):
    assert client.post("/predict", json={"consumption": "abc", "voltage": 230}).status_code == 422
    assert client.post("/predict", json={"consumption": -1, "voltage": 230}).status_code == 422
    assert client.post("/predict", json={"consumption": 1, "voltage": 0}).status_code == 422
    assert client.post("/predict", json={"consumption": 1, "voltage": 230, "meter_id": "NOPE"}).status_code == 404


def test_meters_summary(client):
    rows = client.get("/meters").json()
    assert len(rows) == 5
    assert all(0 <= r["flag_rate"] <= 1 for r in rows)


def test_metrics_endpoint_404_when_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "METRICS_PATH", str(tmp_path / "missing.json"))
    assert client.get("/metrics").status_code == 404
