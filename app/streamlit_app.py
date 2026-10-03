from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------- Paths (relative to this file, not to where you run) ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "test_predictions.csv"
FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Traffic Flow Forecasting", layout="wide")


# ---------- Load data once (cached) ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE, parse_dates=["date_time"])


def show_figure(name, caption):
    #  Show a figure if it exists
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not found: {name}")


df = load_data()

# ----------  Title ----------
st.title("🚗 Traffic Flow Forecasting")
st.write(
    "Hourly traffic volume forecasts for one highway sensor (I-94 westbound, "
    "Minnesota) using XGBoost. All numbers below come from a **held-out test "
    "year (Oct 2017 - Sep 2018)** that was not used during model development."
)

# ----------  Sidebar ----------
st.sidebar.header("Settings")
model_label = st.sidebar.selectbox("Model", ["B: 24 hours ahead", "A: 1 hour ahead"])
pred_col = "pred_b" if model_label.startswith("B") else "pred_a"

min_date = df["date_time"].min().date()
max_date = df["date_time"].max().date()
start_date = st.sidebar.date_input(
    "Start date",
    value=pd.Timestamp("2018-03-05").date(),
    min_value=min_date,
    max_value=max_date,
)
n_days = st.sidebar.slider("Number of days", 1, 14, 7)
show_baseline = st.sidebar.checkbox("Show baseline (hour-of-week average)", value=True)

# ---------- Filter the selected period ----------
start = pd.Timestamp(start_date)
end = start + pd.Timedelta(days=n_days)
view = df[(df["date_time"] >= start) & (df["date_time"] < end)]

tab_forecast, tab_results, tab_analysis = st.tabs(
    ["Forecast", "Final results", "Analysis"]
)

# =================  Tab 1: Forecast =================
with tab_forecast:
    if view.empty:
        st.warning("No data in this period. Choose another start date.")
    else:
        # MAE on the same rows for both
        valid = view.dropna(subset=["traffic_volume", pred_col, "pred_avg"])

        c1, c2, c3 = st.columns(3)
        if valid.empty:
            c1.warning("Not enough rows to compute MAE.")
        else:
            mae_model = (valid["traffic_volume"] - valid[pred_col]).abs().mean()
            mae_base = (valid["traffic_volume"] - valid["pred_avg"]).abs().mean()
            c1.metric(
                "Model MAE (veh/h)",
                f"{mae_model:.0f}",
                delta=f"{mae_model - mae_base:+.0f} vs baseline",
                delta_color="inverse", 
            )
            c2.metric("Baseline MAE (veh/h)", f"{mae_base:.0f}")
            c3.metric("Hours compared", f"{len(valid)}")

        # Plot: actual vs predicted
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(view["date_time"], view["traffic_volume"], label="Actual")
        ax.plot(view["date_time"], view[pred_col], label=f"XGBoost {model_label}")
        if show_baseline:
            ax.plot(
                view["date_time"], view["pred_avg"],
                label="Baseline (hour-of-week avg)", linestyle="--", alpha=0.7,
            )
        ax.set_ylabel("Vehicles per hour")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

# ================= Tab 2: Final results =================
with tab_results:
    st.subheader("Test-set MAE (vehicles per hour)")
    results = pd.DataFrame(
        {
            "Model": [
                "Constant (train mean)",
                "Naive (same hour last week)",
                "Hour-of-week average",
                "XGBoost B (24 hours ahead)",
                "XGBoost A (1 hour ahead)",
            ],
            "MAE": [1730, 343, 276, 225, 150],
            "% of mean traffic": ["52.0%", "10.3%", "8.3%", "6.7%", "4.5%"],
        }
    ).set_index("Model")
    st.table(results)
    st.write(
        "Mean traffic in the test year is about 3,328 vehicles/hour. "
        "XGBoost B improves on the best baseline by about 18.5%. "
        "Model A also uses the previous hours, which the baselines do not, "
        "so B is the fairer comparison."
    )

# ================= Tab 3: Analysis =================
with tab_analysis:
    show_figure("feature_importance.png", "Feature importance (XGBoost B)")
    show_figure("error_by_hour.png", "Mean absolute error by hour (validation)")
    show_figure("actual_vs_pred_week.png", "Actual vs predicted, one validation week")

    with st.expander("Limitations"):
        st.write(
            "- One sensor, one location: results may not transfer to other roads.\n"
            "- Observed weather is used as if the weather forecast were perfect.\n"
            "- No heavy hyperparameter tuning.\n"
            "- Incidents and closures are not predicted; two outage periods were excluded.\n"
            "- Largest errors are during the morning peak and on public holidays."
        )
