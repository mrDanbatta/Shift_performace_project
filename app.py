import streamlit as st
import pandas as pd
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Shift Performance Optimisation", layout="wide")
st.title("Shift Performance Optimisation System")

# predictive input form
st.header("Predictive Insights and Optimisation Recommendations")

response = requests.get(f"{API_URL}/data")
if response.status_code == 200:
    form_data = response.json()

shift_choice = st.selectbox("Select Shift", form_data["shifts"])
col1, col2 = st.columns(2)

with col1:
    supervisor_choice = st.selectbox("Select Supervisor", form_data["supervisors"])
    operators = st.multiselect("Select Operators", form_data["operators"])
    skill_category_choice = st.selectbox("Select Skill Category", form_data["skill_categories"])
    machine_status_choice = st.selectbox("Select Machine Status", form_data["machine_statuses"])
    issue_type_choice = st.selectbox("Select Issue Type", form_data["issue_types"])
    inspection_result_choice = st.selectbox("Select Inspection Result", form_data["inspection_results"])
    maintenance_flag = st.selectbox("Maintenance Flag", [0, 1])
with col2:
    cycle_time = st.slider("Cycle Time (seconds)", 0.0, 300.0, 60.0, step=1.0)
    downtime = st.slider("Maintenance Downtime (hours)", 0.0, 20.0, 2.0, step=0.5) 
    defect_count = st.slider("Defect Count", 0.0, 100.0, 5.0, step=0.1)
    experience = st.slider("Average Operator Experience (years)", 1, 15, 6, step=1)
    temperature = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, step=0.5)
    humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0, step=1.0)

cycle_time_avg = cycle_time * 60/ 800  # convert to minutes per 800 units

if st.button("Predict Shift Efficiency"):
    input_data = {
        "shift_name": shift_choice,
        "supervisor_id": supervisor_choice,
        "defect_count": defect_count,
        "cycle_time_avg": cycle_time_avg,
        "operator_id": operators[0] if operators else 'OP_001',
        "experience_level": experience,
        "skill_category": skill_category_choice,
        "maintenance_downtime": downtime,
        "maintenance_flag": maintenance_flag,
        "machine_status": machine_status_choice,
        "issue_type": issue_type_choice,
        "inspection_result": inspection_result_choice,
        "temperature": temperature,
        "humidity": humidity
    }
    response = requests.post(f"{API_URL}/predict", json=input_data)
    if response.ok:
        prediction = response.json()["predicted_efficiency"]
        st.metric(label="Predicted Shift Efficiency (%)", value=f"{prediction:.2f}%")
    else:
        st.error("Error during prediction. Please try again.")

st.markdown("---")

st.subheader("Shift Scenario Optimisation")

exp_min, exp_max = st.slider("Experience Range (years)", 1, 15, (1, 10), step=1)
downtime_min, downtime_max = st.slider("Downtime Range (hours)", 0.0, 20.0, (0.0, 10.0), step=0.5)
defect_count_min, defect_count_max = st.slider("Defect Count Range", 0.0, 100.0, (0.0, 50.0), step=0.1)

if st.button("Optimise Shift"):
    optimisation_input = {
        "shift_name": shift_choice,
        "exp_range": (exp_min, exp_max),
        "downtime_range": (downtime_min, downtime_max),
        "defect_count_range": (defect_count_min, defect_count_max)
    }
    response = requests.post(f"{API_URL}/optimise", json=optimisation_input)
    if response.ok:
        optimisation_results = pd.DataFrame(response.json())
        st.write("Optimisation Results:")
        st.dataframe(optimisation_results)
    else:
        st.error("Error during optimisation. Please try again.")

st.markdown("---")

if st.button("Retrain Model"):
    response = requests.post(f"{API_URL}/retrain")
    if response.ok:
        st.success("Model is being retrained in the background. Please wait a few moments before making predictions.")
    else:
        st.error("Error during model retraining. Please try again.")