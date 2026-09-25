import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("ETD_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 30

st.set_page_config(page_title="Electricity Theft Detection", layout="wide")


@st.cache_data(ttl=60)
def fetch(path, meter_id=None):
    params = {"meter_id": meter_id} if meter_id else None
    res = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


def post_prediction(payload):
    res = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


st.title("Electricity Theft Detection")
st.caption(
    "Unsupervised anomaly detection on simulated smart-meter data. "
    "Ground-truth theft labels exist only because the data is simulated; the model never sees them."
)

try:
    meters = pd.DataFrame(fetch("/meters"))
except Exception as e:
    st.error(f"Cannot reach the API at {API_URL}: {e}")
    st.info("Start it with: uvicorn backend.main:app --reload")
    st.stop()

overview_tab, meter_tab, model_tab, check_tab = st.tabs(
    ["Overview", "Investigate a meter", "Model performance", "Check a reading"]
)

# Overview
with overview_tab:
    total, flagged = int(meters["readings"].sum()), int(meters["flagged"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Readings scored", f"{total:,}")
    c2.metric("Flagged as suspicious", f"{flagged:,}")
    c3.metric("Flag rate", f"{flagged / total:.1%}")

    st.subheader("Flag rate by meter")
    st.altair_chart(
        alt.Chart(meters)
        .mark_bar()
        .encode(
            x=alt.X("meter_id:N", title="Meter"),
            y=alt.Y("flag_rate:Q", title="Share of readings flagged", axis=alt.Axis(format="%")),
            tooltip=["meter_id", "readings", "flagged", alt.Tooltip("flag_rate:Q", format=".1%")],
        ),
        width="stretch",
    )
    st.dataframe(meters, width="stretch", hide_index=True)

# Meter drill-down
with meter_tab:
    meter_id = st.selectbox("Meter", meters["meter_id"].tolist())
    df = pd.DataFrame(fetch("/detect", meter_id))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["flagged"] = df["anomaly"] == -1
    df["outcome"] = None
    df.loc[df["flagged"] & (df["is_theft"] == 1), "outcome"] = "Caught theft"
    df.loc[df["flagged"] & (df["is_theft"] == 0), "outcome"] = "False alarm"
    df.loc[~df["flagged"] & (df["is_theft"] == 1), "outcome"] = "Missed theft"

    base = alt.Chart(df).encode(x=alt.X("timestamp:T", title="Time"))
    usage = base.mark_line(color="#4c78a8").encode(y=alt.Y("consumption:Q", title="Consumption (kWh)"))
    expected = base.mark_line(color="#888", strokeDash=[4, 4]).encode(y="expected_consumption:Q")
    marks = (
        base.transform_filter("datum.outcome != null")
        .mark_point(filled=True, size=60)
        .encode(
            y="consumption:Q",
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["Caught theft", "False alarm", "Missed theft"],
                    range=["#d62728", "#ff9f1c", "#7f7f7f"],
                ),
                title=None,
            ),
            tooltip=["timestamp:T", "consumption", "expected_consumption", "zscore", "outcome"],
        )
    )
    st.altair_chart((usage + expected + marks).interactive(), width="stretch")
    st.caption("Solid line: actual usage. Dashed line: expected usage for that hour (median of the previous 7 days).")

    st.subheader("Most suspicious readings")
    cases = pd.DataFrame(fetch("/fraud-cases", meter_id))
    if cases.empty:
        st.success("No readings flagged for this meter.")
    else:
        st.dataframe(
            cases[["timestamp", "consumption", "expected_consumption", "zscore", "anomaly_score", "theft_type"]].head(25),
            width="stretch",
            hide_index=True,
        )

# Model performance
with model_tab:
    try:
        metrics = fetch("/metrics")
    except Exception as e:
        st.warning(f"No evaluation results available ({e}). Run: python -m model.evaluate")
        metrics = None

    if metrics:
        data, models = metrics["dataset"], metrics["models"]
        st.markdown(
            f"Trained on the first {metrics['config']['train_fraction']:.0%} of the timeline "
            f"({data['train_rows']:,} readings, no labels) and evaluated on the later "
            f"{data['test_rows']:,} readings, of which {data['test_theft_rate']:.1%} are theft. "
            f"A model flagging at random would score PR-AUC of about {data['test_theft_rate']:.2f}."
        )

        table = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "F1": r["f1"],
                    "ROC-AUC": r["roc_auc"],
                    "PR-AUC": r["pr_auc"],
                    "Fit time (s)": r["fit_seconds"],
                }
                for name, r in models.items()
            ]
        )
        st.subheader("Model comparison")
        st.dataframe(table.round(3), width="stretch", hide_index=True)

        deployed = metrics["deployed_model"]
        st.subheader(f"Confusion matrix: {deployed} (deployed)")
        cm = models[deployed]["confusion"]
        st.dataframe(
            pd.DataFrame(
                {"Predicted normal": [cm["tn"], cm["fn"]], "Predicted theft": [cm["fp"], cm["tp"]]},
                index=["Actually normal", "Actually theft"],
            ),
            width="stretch",
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Recall by theft type")
            st.dataframe(
                pd.DataFrame({n: r["recall_by_type"] for n, r in models.items()}).T.round(2),
                width="stretch",
            )
        with right:
            st.subheader("Threshold sensitivity (Isolation Forest)")
            sweep = pd.DataFrame(metrics["contamination_sweep"]).melt(
                "contamination", ["precision", "recall", "f1"], "metric", "value"
            )
            st.altair_chart(
                alt.Chart(sweep)
                .mark_line(point=True)
                .encode(x="contamination:Q", y="value:Q", color="metric:N"),
                width="stretch",
            )

        st.subheader(f"Stability across {len(metrics['robustness']['seeds'])} independently simulated datasets")
        rob = pd.DataFrame(
            [
                {
                    "Model": name,
                    "F1": f"{r['f1']['mean']:.3f} +/- {r['f1']['std']:.3f}",
                    "ROC-AUC": f"{r['roc_auc']['mean']:.3f} +/- {r['roc_auc']['std']:.3f}",
                    "PR-AUC": f"{r['pr_auc']['mean']:.3f} +/- {r['pr_auc']['std']:.3f}",
                }
                for name, r in metrics["robustness"]["models"].items()
            ]
        )
        st.dataframe(rob, width="stretch", hide_index=True)
        st.caption("All numbers are from simulated data and describe this simulator, not real utility data.")

# Score a new reading
with check_tab:
    st.write(
        "Pick a meter to score the reading against that meter's own history. "
        "With no meter selected the reading is judged in isolation, which is much weaker."
    )
    choice = st.selectbox("Meter (optional)", ["(none)"] + meters["meter_id"].tolist())
    col1, col2 = st.columns(2)
    consumption = col1.number_input("Consumption (kWh)", min_value=0.0, value=1.0, step=0.1)
    voltage = col2.number_input("Voltage (V)", min_value=1.0, value=230.0, step=1.0)

    if st.button("Run detection"):
        payload = {"consumption": consumption, "voltage": voltage}
        if choice != "(none)":
            payload["meter_id"] = choice
        try:
            result = post_prediction(payload)
            if result["is_fraud"]:
                st.error("Suspected theft")
            else:
                st.success("Normal usage")
            m1, m2 = st.columns(2)
            m1.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
            m2.metric("Scored against", "meter history" if result["context"] == "history" else "reading alone")
            with st.expander("Full response"):
                st.json(result)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
