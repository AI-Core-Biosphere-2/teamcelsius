# # app.py
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# from simulation import train_var_model, simulate_scenario
# from ai_integration import generate_summary
# from experts import add_expert_message, generate_expert_response, get_conversation_log

# st.set_page_config(page_title="Ecosystem Simulation & Collaborative Forecasting", layout="wide")

# st.title("Ecosystem Simulation and Collaborative Forecasting")

# # 1. Data Upload and Summary
# st.header("Data Overview")
# @st.cache_data
# def load_merged_data(filepath="merged_data.csv"):
#     data = pd.read_csv(filepath, parse_dates=["DateTime"])
#     return data

# merged_data = load_merged_data()

# # Filter for RainForest MountainTower data.
# # Assuming sensor columns include 'RainForest_MountainTower_' in their names.
# df_rf = merged_data[merged_data["Location"] == "RainForest"]
# sensor_cols = [col for col in df_rf.columns if "MountainTower" in col]
# if not sensor_cols:
#     st.error("No MountainTower sensor data found in RainForest location.")
# else:
#     df_mt = df_rf[["DateTime", "Location"] + sensor_cols]
#     st.write("MountainTower Data Preview:", df_mt.head())

#     # 2. Ecosystem Simulation
#     st.header("Ecosystem Simulation")
#     st.markdown("Adjust simulation parameters below and run a forecast simulation for the next 24 hours.")

#     # Sidebar inputs for simulation adjustments.
#     st.sidebar.subheader("Simulation Settings")
#     temp_adjust = st.sidebar.slider("Temperature Change (°C)", -5.0, 5.0, 0.0, 0.5)
#     wind_adjust = st.sidebar.slider("Wind Speed Change (m/s)", -5.0, 5.0, 0.0, 0.5)
#     run_simulation = st.sidebar.button("Run Simulation")

#     # Prepare the MountainTower dataset for simulation:
#     # We use only the sensor columns of interest.
#     simulation_cols = sensor_cols  # e.g., Temperature, RH, radiation, wind, PAR, global rad.
#     simulation_data = df_mt[["DateTime"] + simulation_cols].dropna()
    
#     if run_simulation:
#         st.info("Training VAR model (this may take a moment)...")
#         var_results = train_var_model(simulation_data)
#         st.success("VAR model trained successfully.")
        
#         # Define adjustments: keys must match the column names exactly.
#         adjustments = {}
#         # For demonstration, assume the temperature column contains 'Temp' in its name.
#         temp_cols = [col for col in simulation_cols if "Temp" in col]
#         wind_cols = [col for col in simulation_cols if "WindSpeed" in col or "wind" in col]
#         if temp_cols:
#             adjustments[temp_cols[0]] = temp_adjust
#         if wind_cols:
#             adjustments[wind_cols[0]] = wind_adjust
        
#         st.write("Adjustments applied:", adjustments)
#         forecast_df = simulate_scenario(simulation_data, var_results, adjustments, steps=24)
        
#         st.subheader("Forecast for Next 24 Hours")
#         st.line_chart(forecast_df)

#         # Generate natural language summary using AI.
#         simulation_text = (f"Temperature adjusted by {temp_adjust}°C and Wind Speed adjusted by {wind_adjust} m/s. "
#                            "Forecast shows the impact on related variables.")
#         summary = generate_summary(simulation_text)
#         st.subheader("AI-Generated Simulation Summary")
#         st.write(summary)

#     # 3. Collaborative Forecasting and AI-to-AI Knowledge Exchange
#     st.header("Collaborative Forecasting (AI Experts)")
#     st.markdown("Simulated expert discussion among AI specialists analyzing different variables.")

#     # Display the conversation log.
#     log = get_conversation_log()
#     if log:
#         for entry in log:
#             st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
#     else:
#         st.info("No expert messages yet.")

#     # Button to simulate an expert discussion round.
#     if st.button("Generate Expert Discussion"):
#         # For example, Temperature Expert and Humidity Expert exchange insights.
#         add_expert_message("Temperature Expert", "I observe that the temperature trend suggests a moderate rise tomorrow.")
#         context = "Temperature Expert: I observe that the temperature trend suggests a moderate rise tomorrow."
#         humidity_response = generate_expert_response("Humidity Expert", context)
#         st.success("Expert discussion updated.")
#         # Refresh and display updated log.
#         log = get_conversation_log()
#         for entry in log:
#             st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")


# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from simulation import train_var_model, simulate_scenario
from ai_integration import generate_summary
from experts import add_expert_message, generate_expert_response, get_conversation_log

st.set_page_config(page_title="Ecosystem Simulation & Collaborative Forecasting", layout="wide")
st.title("Ecosystem Simulation and Collaborative Forecasting")

# 1. Data Upload and Summary
st.header("Data Overview")

@st.cache_data
def load_merged_data(filepath="merged_data.csv"):
    data = pd.read_csv(filepath, parse_dates=["DateTime"])
    return data

merged_data = load_merged_data()

# --- Ecosystem Selection ---
ecosystem_options = merged_data["Location"].unique().tolist()
selected_ecosystem = st.sidebar.selectbox("Select Ecosystem", ecosystem_options)
df_ecosystem = merged_data[merged_data["Location"] == selected_ecosystem]

# For simulation, we use all sensor columns (all except DateTime and Location).
sensor_cols = [col for col in df_ecosystem.columns if col not in ["DateTime", "Location"]]

if not sensor_cols:
    st.error(f"No sensor data found for {selected_ecosystem}.")
else:
    df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
    st.write(f"Data Preview for {selected_ecosystem}:", df_sim.head())

    # 2. Ecosystem Simulation
    st.header("Ecosystem Simulation")
    st.markdown("Adjust simulation parameters below and run a forecast simulation for the next 24 hours.")

    # Sidebar inputs for simulation adjustments.
    st.sidebar.subheader("Simulation Settings")
    # Here you can adjust sliders; these should be set to reasonable values.
    temp_adjust = st.sidebar.slider("Temperature Change (°C)", -1.0, 1.0, 0.0, 0.05)
    wind_adjust = st.sidebar.slider("Wind Speed Change (m/s)", -1.0, 1.0, 0.0, 0.05)
    run_simulation = st.sidebar.button("Run Simulation")

    # For simulation, drop rows with missing values
    simulation_data = df_sim[["DateTime"] + sensor_cols].dropna()

    if run_simulation:
        st.info("Training VAR model (this may take a moment)...")
        try:
            var_results = train_var_model(simulation_data)
        except Exception as e:
            st.error(f"VAR model training failed: {e}. Try using smaller adjustments or check the input data.")
        else:
            st.success("VAR model trained successfully.")
            
            # Prepare adjustments. Here, we try to match variables by looking for keywords.
            adjustments = {}
            # For example, if a column contains "Temp", adjust it.
            temp_cols = [col for col in sensor_cols if "Temp" in col or "temp" in col]
            wind_cols = [col for col in sensor_cols if "Wind" in col or "wind" in col]
            if temp_cols:
                adjustments[temp_cols[0]] = temp_adjust
            if wind_cols:
                adjustments[wind_cols[0]] = wind_adjust

            st.write("Adjustments applied:", adjustments)
            try:
                forecast_df = simulate_scenario(simulation_data, var_results, adjustments, steps=24)
            except Exception as sim_err:
                st.error(f"Simulation failed: {sim_err}. Consider reducing the adjustment values.")
            else:
                st.subheader("Forecast for Next 24 Hours")
                st.line_chart(forecast_df)

                # Generate natural language summary using AI.
                simulation_text = (f"Temperature adjusted by {temp_adjust}°C and Wind Speed adjusted by {wind_adjust} m/s in {selected_ecosystem}. "
                                   "Forecast shows the impact on related variables.")
                summary = generate_summary(simulation_text)
                st.subheader("AI-Generated Simulation Summary")
                st.write(summary)

    # 3. Collaborative Forecasting and AI-to-AI Knowledge Exchange
    st.header("Collaborative Forecasting (AI Experts)")
    st.markdown("Simulated expert discussion among AI specialists analyzing different variables.")

    # Display the conversation log.
    log = get_conversation_log()
    if log:
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
    else:
        st.info("No expert messages yet.")

    # Button to simulate an expert discussion round.
    if st.button("Generate Expert Discussion"):
        # For demonstration, Temperature and Humidity experts exchange insights.
        add_expert_message("Temperature Expert", "I observe that the temperature trend suggests a moderate rise tomorrow.")
        context = "Temperature Expert: I observe that the temperature trend suggests a moderate rise tomorrow."
        humidity_response = generate_expert_response("Humidity Expert", context)
        st.success("Expert discussion updated.")
        # Refresh and display updated log.
        log = get_conversation_log()
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
