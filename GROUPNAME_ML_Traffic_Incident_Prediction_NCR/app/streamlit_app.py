"""Streamlit app: predict yearly vehicular crashes on an NCR road.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config as C  # noqa: E402
import inference as I  # noqa: E402

st.set_page_config(page_title="NCR Road Crash Predictor", page_icon=":traffic_light:", layout="wide")


@st.cache_resource
def get_artifacts():
    return I.load_artifacts()


@st.cache_data
def get_data():
    return pd.read_csv(C.PROCESSED_CSV)


try:
    model, meta = get_artifacts()
    data = get_data()
except FileNotFoundError:
    st.error("Model or data file not found. Run `python src/data_prep.py` and "
             "`python src/train.py` first (see README).")
    st.stop()

ranges = meta["training_ranges"]
test_mae = meta["final_test_metrics"]["MAE"]

st.title("NCR Road Crash Predictor")
st.caption(f"Model: **{meta['model_name']}** | trained on {meta['train_years'][0]}-"
           f"{meta['train_years'][1]} MMDA AADT and MMARAS data | decision-support tool only")

tab_form, tab_csv, tab_about = st.tabs(["Single prediction", "Batch CSV upload", "About the model"])

with tab_form:
    st.subheader("Estimate yearly crashes for one road")
    roads = ["(enter values manually)"] + sorted(data["road"].unique())
    choice = st.selectbox("Start from an existing road and year (optional)", roads)
    defaults = {"aadt": 150000.0, "prev_crashes": 2000.0, "lane_km": 100.0, "year": 2025}
    if choice != roads[0]:
        year_opts = sorted(data.loc[data["road"] == choice, "year"].unique())
        pick = st.selectbox("Year of the record", year_opts, index=len(year_opts) - 1)
        row = data[(data["road"] == choice) & (data["year"] == pick)].iloc[0]
        defaults = {"aadt": float(row["aadt"]),
                    "prev_crashes": float(row["prev_crashes"]) if pd.notna(row["prev_crashes"]) else 0.0,
                    "lane_km": float(row["lane_km"]), "year": int(pick)}
    c1, c2 = st.columns(2)
    aadt = c1.number_input("AADT: average vehicles per day", value=defaults["aadt"], step=1000.0,
                           help="Annual Average Daily Traffic from the MMDA counting stations.")
    prev = c2.number_input("Crashes recorded in the previous year", value=defaults["prev_crashes"],
                           step=10.0, help="MMARAS crash count for the same road, one year earlier.")
    lane_km = c1.number_input("Lane-kilometers (road length x average lanes)",
                              value=defaults["lane_km"], step=10.0)
    year = c2.number_input("Year to predict", value=defaults["year"], step=1, format="%d")

    if st.button("Predict crashes", type="primary"):
        frame = pd.DataFrame([{"aadt": aadt, "prev_crashes": prev, "lane_km": lane_km, "year": year}])
        work, problems = I.validate_frame(frame, ranges)
        errors = [m for m in problems.get(0, []) if m.startswith("ERROR")]
        warnings = [m for m in problems.get(0, []) if m.startswith("WARNING")]
        if errors:
            for m in errors:
                st.error(m.replace("ERROR: ", ""))
        else:
            for m in warnings:
                st.warning(m.replace("WARNING: ", ""))
            pred = float(I.predict_frame(model, work, problems)["predicted_crashes"].iloc[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted crashes", f"{pred:,.0f}")
            m2.metric("Typical error (test MAE)", f"± {test_mae:,.0f}")
            m3.metric("Change vs previous year", f"{pred - prev:+,.0f}")
            st.caption("This model returns a single estimate and does not provide probabilities. "
                       "The typical error is the average miss on the 2023-2024 test years.")

with tab_csv:
    st.subheader("Predict for many road-years")
    st.write("Upload a CSV with columns `aadt`, `prev_crashes`, `lane_km`, `year` "
             "(an optional `road` column is kept in the output).")
    template = pd.DataFrame({"road": ["EXAMPLE ROAD"], "aadt": [150000], "prev_crashes": [2000],
                             "lane_km": [100.0], "year": [2025]})
    st.download_button("Download CSV template", template.to_csv(index=False), "template.csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            if raw.empty:
                raise ValueError("The uploaded file has no rows.")
            work, problems = I.validate_frame(raw, ranges)
            res = I.predict_frame(model, work, problems)
            bad = sum(any(m.startswith("ERROR") for m in v) for v in problems.values())
            st.success(f"{len(res) - bad} of {len(res)} rows predicted; {bad} rejected.") if bad < len(res) \
                else st.error("No valid rows to predict.")
            st.dataframe(res)
            st.download_button("Download results", res.to_csv(index=False), "predictions.csv")
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("The file could not be read as a CSV. Check the format and try again.")

with tab_about:
    st.subheader("How the model was chosen")
    st.write(f"Three algorithms were compared: Linear Regression, Random Forest and Gradient Boosting. "
             f"**{meta['model_name']}** had the lowest validation MAE (primary metric) and was scored "
             f"once on the untouched 2023-2024 test years.")
    val = pd.DataFrame(meta["validation"])[["model", "CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_R2_mean"]]
    st.dataframe(val.round(2), hide_index=True)
    ft = meta["final_test_metrics"]
    st.write(f"Final test: MAE {ft['MAE']:,.0f} | RMSE {ft['RMSE']:,.0f} | R² {ft['R2']:.3f} | "
             f"MAPE {ft['MAPE_%']:.1f}%")
    nb = meta["naive_baseline_test_metrics"]
    st.info(f"Honest reference: simply repeating last year's count scored MAE {nb['MAE']:,.0f} on the "
            f"same test years, so the model does not beat that simple baseline.")
    st.subheader("Inputs")
    st.table(pd.DataFrame({"Feature": [C.FEATURE_LABELS[f] for f in C.FEATURES],
                           "Training range": [f"{ranges[f][0]:,.0f} to {ranges[f][1]:,.0f}" for f in C.FEATURES]}))
    st.subheader("Limitations")
    st.markdown("- Yearly road totals only; not a real-time or per-incident predictor.\n"
                "- Trained on 14 roads; tree models cannot extrapolate beyond the training range "
                "(e.g. a road far larger than EDSA).\n"
                "- Excludes weather, driver behaviour and road geometry.")
