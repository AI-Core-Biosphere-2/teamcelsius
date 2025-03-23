# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from alternative_models import forecast_arima, forecast_prophet
from ai_integration import generate_summary
from experts import add_expert_message, generate_expert_response, get_conversation_log

st.set_page_config(page_title="Ecosystem Simulation & Collaborative Forecasting", layout="wide")
st.title("Ecosystem Simulation and Collaborative Forecasting")

# Load merged data
@st.cache_data
def load_merged_data(filepath="merged_data.csv"):
    data = pd.read_csv(filepath, parse_dates=["DateTime"])
    return data

merged_data = load_merged_data()

# Sidebar: Page and Ecosystem selection
page = st.sidebar.radio("Select Page", ["Dashboard", "Forecasting", "Visualizations", "Expert Collaboration"])
ecosystem_options = merged_data["Location"].unique().tolist()
selected_ecosystem = st.sidebar.selectbox("Select Ecosystem", ecosystem_options)
df_ecosystem = merged_data[merged_data["Location"] == selected_ecosystem]
sensor_cols = [col for col in df_ecosystem.columns if col not in ["DateTime", "Location"]]

if page == "Dashboard":
    st.header("Dashboard")
    st.write(f"Data Preview for {selected_ecosystem}:")
    if sensor_cols:
        df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
        st.dataframe(df_sim.head(10))
    else:
        st.error(f"No sensor data found for {selected_ecosystem}.")

elif page == "Forecasting":
    st.header("Forecasting")
    if not sensor_cols:
        st.error(f"No sensor data found for {selected_ecosystem}.")
    else:
        df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
        st.write(f"Data Preview for {selected_ecosystem}:", df_sim.head())
        # Prepare simulation data: set DateTime as index for time interpolation.
        raw_simulation_data = df_sim[["DateTime"] + sensor_cols]
        simulation_data = raw_simulation_data.copy()
        simulation_data = simulation_data.set_index("DateTime")
        simulation_data[sensor_cols] = simulation_data[sensor_cols].interpolate(method='time').ffill().bfill()
        simulation_data = simulation_data.reset_index()
        st.write("Simulation data shape after interpolation:", simulation_data.shape)
        if simulation_data.shape[0] < 5:
            st.error("Not enough data available for forecasting. Please check your dataset.")
        else:
            st.sidebar.subheader("Forecast Model")
            forecast_model = st.sidebar.selectbox("Select Forecast Model", ["ARIMA", "Prophet"])
            st.sidebar.subheader("Simulation Settings")
            temp_adjust = st.sidebar.slider("Temperature Change (°F)", -5.0, 5.0, 0.0, 0.5)
            wind_adjust = st.sidebar.slider("Wind Speed Change (m/s)", -5.0, 5.0, 0.0, 0.5)
            run_simulation = st.sidebar.button("Run Simulation")
            forecast_df = None
            if run_simulation:
                if forecast_model == "ARIMA":
                    st.info("Forecasting with ARIMA model...")
                    try:
                        forecast_df = forecast_arima(simulation_data, order=(1,1,1), steps=24)
                    except Exception as e:
                        st.error(f"ARIMA forecast failed: {e}")
                elif forecast_model == "Prophet":
                    st.info("Forecasting with Prophet model...")
                    try:
                        forecast_df = forecast_prophet(simulation_data, steps=24)
                    except Exception as e:
                        st.error(f"Prophet forecast failed: {e}")
                if forecast_df is not None:
                    st.subheader("Forecast for Next 24 Hours")
                    st.line_chart(forecast_df)
                    simulation_text = (f"In {selected_ecosystem}, Temperature adjusted by {temp_adjust}°F and Wind Speed by {wind_adjust} m/s. "
                                       f"Forecast using {forecast_model} shows the impact on related variables.")
                    summary = generate_summary(simulation_text)
                    st.subheader("AI-Generated Simulation Summary")
                    st.write(summary)

elif page == "Visualizations":
    st.header("Visualizations")
    viz_page = st.sidebar.radio("Select Visualization", ["Correlation Heatmap", "Metric Variation Over Time"])
    df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
    if viz_page == "Correlation Heatmap":
        st.subheader("Sensor Data Correlation")
        corr_matrix = df_sim[sensor_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    elif viz_page == "Metric Variation Over Time":
        st.subheader("Metric Variation Over Time")
        selected_metric = st.selectbox("Select Metric", sensor_cols)
        fig_line = px.line(df_sim, x="DateTime", y=selected_metric, title=f"{selected_metric} Over Time")
        st.plotly_chart(fig_line, use_container_width=True)

elif page == "Expert Collaboration":
    st.header("Expert Collaboration (AI Experts)")
    st.markdown("Simulated expert discussion among AI specialists analyzing different variables.")
    log = get_conversation_log()
    if log:
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
    else:
        st.info("No expert messages yet.")
    if st.button("Generate Expert Discussion"):
        add_expert_message("Temperature Expert", "Based on the latest sensor data, I observe a subtle upward trend in temperature.")
        context = ("Temperature Expert: I observe a subtle upward trend in temperature which might affect humidity and other related variables. "
                   "Please analyze how this trend could impact overall ecosystem dynamics.")
        generate_expert_response("Humidity Expert", context)
        st.success("Expert discussion updated.")
        log = get_conversation_log()
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
