"""Streamlit app: predict yearly vehicular crashes on an NCR road.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
import inference as I  # noqa: E402

st.set_page_config(page_title="NCR Road Crash Predictor", page_icon=":traffic_light:", layout="wide")

# White background and black text theme
st.markdown(
    """
    <style>
    /* White background and black text */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    header, footer {
        background-color: #FFFFFF !important;
    }
    /* Headings and labels */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    p, span, label {
        color: #111111;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    /* Action / Submit Buttons: solid black with crisp bold WHITE text */
    .stButton > button,
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
    }
    .stButton > button *,
    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover,
    button[kind="primary"]:hover {
        background-color: #333333 !important;
        border-color: #333333 !important;
    }
    .stButton > button:hover *,
    .stButton > button:hover p {
        color: #FFFFFF !important;
    }

    /* Download Buttons: clean white with black border and black text */
    .stDownloadButton > button,
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span,
    button[kind="secondary"] *,
    button[kind="secondary"] p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #F4F4F5 !important;
        border-color: #000000 !important;
    }
    .stDownloadButton > button:hover *,
    .stDownloadButton > button:hover p {
        color: #000000 !important;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #444444 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        border-bottom-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

tab_form, tab_rec, tab_csv, tab_about = st.tabs([
    "Single prediction",
    "Recursive multi-year forecast",
    "Batch CSV upload",
    "About the model",
])

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
                       "The typical error is the average miss on the 2023-2024 test years. "
                       "To forecast further years where this estimate feeds into the next, "
                       "switch to the **Recursive multi-year forecast** tab.")

with tab_rec:
    st.subheader("Iterative multi-year forecast (chain prediction)")
    st.write(
        "Forecast crashes sequentially across multiple future years. The model predicts Year 1 "
        "using your baseline, then **recursively feeds that predicted value** as `prev_crashes` "
        "for Year 2, Year 3, and beyond."
    )

    rec_roads = ["(enter values manually)"] + sorted(data["road"].unique())
    rec_choice = st.selectbox("Start from an existing road (optional)", rec_roads, key="rec_road_choice")

    rec_defaults = {"aadt": 150000.0, "prev_crashes": 2000.0, "lane_km": 100.0, "start_year": 2025}
    if rec_choice != rec_roads[0]:
        road_data = data[data["road"] == rec_choice].sort_values("year")
        latest_row = road_data.iloc[-1]
        rec_defaults = {
            "aadt": float(latest_row["aadt"]),
            # For 2025 forecast, the prior year is 2024 crashes:
            "prev_crashes": float(latest_row["crashes"]) if pd.notna(latest_row["crashes"]) else 0.0,
            "lane_km": float(latest_row["lane_km"]),
            "start_year": int(latest_row["year"]) + 1,
        }

    r1, r2 = st.columns(2)
    start_year = r1.number_input(
        "First forecast year",
        value=rec_defaults["start_year"],
        step=1,
        format="%d",
        key="rec_start_yr",
        help="The initial year to predict.",
    )
    horizon = r2.slider(
        "Forecast horizon (number of years)",
        min_value=1,
        max_value=6,
        value=3,
        key="rec_horizon",
        help="How many consecutive years to project recursively.",
    )

    r3, r4 = st.columns(2)
    rec_prev = r3.number_input(
        f"Baseline crashes recorded in {start_year - 1}",
        value=rec_defaults["prev_crashes"],
        step=10.0,
        key="rec_prev_input",
        help=f"The actual crashes from {start_year - 1} that initialize Step 1.",
    )
    rec_aadt = r4.number_input(
        f"AADT in {start_year} (vehicles/day)",
        value=rec_defaults["aadt"],
        step=1000.0,
        key="rec_aadt_input",
        help="Starting annual average daily traffic.",
    )

    r5, r6 = st.columns(2)
    rec_lane_km = r5.number_input(
        "Lane-kilometers",
        value=rec_defaults["lane_km"],
        step=10.0,
        key="rec_lane_input",
    )
    aadt_growth = r6.slider(
        "Annual traffic (AADT) change (%)",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        key="rec_growth_input",
        help="Annual traffic change rate applied across the forecast period (0% = constant traffic).",
    )

    if st.button("Generate recursive forecast", type="primary", key="btn_run_rec"):
        fc_df = I.recursive_forecast(
            model=model,
            start_year=start_year,
            horizon_years=horizon,
            initial_prev_crashes=rec_prev,
            initial_aadt=rec_aadt,
            lane_km=rec_lane_km,
            aadt_growth_pct=aadt_growth,
            ranges=ranges,
        )

        has_error = fc_df["notes"].str.startswith("ERROR").any()
        if has_error:
            for _, r in fc_df[fc_df["notes"].str.startswith("ERROR")].iterrows():
                st.error(f"Year {int(r['year'])}: {r['notes']}")
        else:
            y1_val = fc_df.iloc[0]["predicted_crashes"]
            yN_val = fc_df.iloc[-1]["predicted_crashes"]
            total_change = yN_val - rec_prev

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(f"{int(fc_df.iloc[0]['year'])} forecast", f"{y1_val:,.0f}")
            col_m2.metric(f"{int(fc_df.iloc[-1]['year'])} forecast", f"{yN_val:,.0f}")
            col_m3.metric(f"Net change vs {start_year - 1}", f"{total_change:+,.0f}")

            # Plot forecast curve
            fig, ax = plt.subplots(figsize=(7, 2.8))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            years_list = fc_df["year"].astype(int).tolist()
            crashes_list = fc_df["predicted_crashes"].tolist()
            ax.plot(years_list, crashes_list, marker='o', color='#000000', linewidth=2, markersize=6)
            for yr_pt, cr_pt in zip(years_list, crashes_list):
                ax.annotate(f"{cr_pt:,.0f}", (yr_pt, cr_pt), textcoords="offset points",
                            xytext=(0, 7), ha='center', fontsize=9, fontweight='bold', color='#000000')
            ax.set_xlabel("Year", color="#000000", fontsize=10)
            ax.set_ylabel("Predicted Crashes", color="#000000", fontsize=10)
            ax.set_xticks(years_list)
            ax.tick_params(colors="#000000")
            ax.grid(True, linestyle='--', alpha=0.25, color="#888888")
            for sp in ax.spines.values():
                sp.set_color('#333333')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Display table
            disp = fc_df[[
                "year", "aadt", "prev_crashes", "prev_crashes_type",
                "predicted_crashes", "change_vs_prev", "notes"
            ]].copy()
            disp.columns = [
                "Year", "AADT (vehicles/day)", "Prev-Year Crashes Used",
                "Input Source", "Predicted Crashes", "Change vs Prev Year", "Notes"
            ]
            disp["AADT (vehicles/day)"] = disp["AADT (vehicles/day)"].map("{:,.0f}".format)
            disp["Prev-Year Crashes Used"] = disp["Prev-Year Crashes Used"].map("{:,.0f}".format)
            disp["Predicted Crashes"] = disp["Predicted Crashes"].map("{:,.0f}".format)
            disp["Change vs Prev Year"] = disp["Change vs Prev Year"].map("{:+,.0f}".format)

            st.dataframe(disp, hide_index=True)

            st.download_button(
                "Download forecast CSV",
                fc_df.to_csv(index=False),
                file_name=f"recursive_forecast_{start_year}_{start_year + horizon - 1}.csv",
                key="btn_dl_rec",
            )

            st.info(
                "💡 **Recursive forecasting mechanism & caveats:**\n\n"
                f"- **Step 1 ({start_year}):** Uses your entered baseline crashes ({rec_prev:,.0f}) from {start_year - 1}.\n"
                "- **Step 2 and beyond:** Automatically inputs the previous step's forecast as `prev_crashes`.\n"
                "- **Compounded uncertainty:** Since each prediction relies on a previous estimated value, "
                "error bounds widen with each future year (typical test error is ±406 crashes per step).\n"
                "- **Tree plateau:** Gradient Boosting models may level off if values approach or exceed the training boundaries."
            )


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
