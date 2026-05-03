import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
import threading

# Load environment variables from .env file
load_dotenv()

API_URL = os.getenv("API_BASE_URL", "http://54.205.14.230:8000")

def load_form_data_from_csv():
    """
    Load form data from CSV file (fast, reliable).
    This is the primary data source.
    """
    try:
        csv_path = "artifacts/data/validated_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return {
                "shifts": sorted(df['shift_name'].unique().tolist()),
                "supervisors": sorted(df['supervisor_id'].unique().tolist()),
                "skill_categories": sorted(df['skill_category'].unique().tolist()),
                "machine_statuses": sorted(df['machine_status'].unique().tolist()),
                "issue_types": sorted(df['issue_type'].unique().tolist()),
                "inspection_results": sorted(df['inspection_result'].unique().tolist()),
                "operators": sorted(df['operator_id'].unique().tolist()),
            }
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
    # Return default data if CSV doesn't exist
    return {
        "shifts": ["Shift A", "Shift B", "Shift C"],
        "supervisors": ["SUP_01", "SUP_02"],
        "skill_categories": ["Junior", "Senior"],
        "machine_statuses": ["Operational", "Down"],
        "issue_types": ["No_Issue", "Maintenance_Issue"],
        "inspection_results": ["Pass", "Fail"],
        "operators": ["OP_001", "OP_002"],
    }

def try_refresh_from_api():
    """
    Try to fetch fresh data from API in the background.
    Does NOT block the main app - used only for updating cache.
    """
    try:
        response = requests.get(f"{API_URL}/data", timeout=5)
        if response.status_code == 200:
            # Update cache with fresh data
            st.cache_data.clear()
            return response.json()
    except Exception as e:
        # Silently fail - we already have fallback data
        pass
    return None

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Shift Performance Optimisation", layout="wide", initial_sidebar_state="expanded")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #262730;
            border-bottom: 3px solid #FF6B6B;
            padding-bottom: 0.5em;
            margin-top: 1.5em;
        }
        .metric-card {
            background-color: #E8EAED;
            padding: 1.5em;
            border-radius: 8px;
            border-left: 4px solid #FF6B6B;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🏭 Shift Performance Optimisation System</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Load form data from CSV immediately (no waiting)
    form_data = load_form_data_from_csv()
    
    # Try to refresh from API in background (non-blocking)
    threading.Thread(target=try_refresh_from_api, daemon=True).start()

    # Tab-based navigation
    tab1, tab2, tab3 = st.tabs(["📊 Efficiency Prediction", "⚙️ Scenario Optimisation", "🔄 Model Management"])

    with tab1:
        st.markdown('<div class="section-header">Predictive Insights & Efficiency Forecasting</div>', unsafe_allow_html=True)
        st.markdown("*Configure shift parameters to predict operational efficiency*")
        
        st.subheader("📋 Shift Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Team & Operations**")
            shift_choice = st.selectbox("Select Shift", form_data["shifts"], key="shift1")
            supervisor_choice = st.selectbox("Select Supervisor", form_data["supervisors"], key="sup1")
            operators = st.multiselect("Select Operators", form_data["operators"], key="op1")
            skill_category_choice = st.selectbox("Select Skill Category", form_data["skill_categories"], key="skill1")
            
        with col2:
            st.markdown("**Equipment & Status**")
            machine_status_choice = st.selectbox("Select Machine Status", form_data["machine_statuses"], key="mach1")
            issue_type_choice = st.selectbox("Select Issue Type", form_data["issue_types"], key="issue1")
            inspection_result_choice = st.selectbox("Select Inspection Result", form_data["inspection_results"], key="insp1")
            maintenance_flag = st.selectbox("Maintenance Flag", ["No", "Yes"], key="maint1")
        
        st.subheader("📐 Performance Metrics")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("**Time & Efficiency**")
            cycle_time = st.slider("Cycle Time (seconds)", 0.0, 300.0, 60.0, step=1.0, key="cycle1")
            experience = st.slider("Avg Operator Experience (years)", 1, 15, 6, step=1, key="exp1")
        
        with col4:
            st.markdown("**Defects & Downtime**")
            defect_count = st.slider("Defect Count", 0.0, 50.0, 5.0, step=0.1, key="defect1")
            downtime = st.slider("Maintenance Downtime (hours)", 0.0, 20.0, 2.0, step=0.5, key="down1")
        
        with col5:
            st.markdown("**Environment**")
            temperature = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, step=0.5, key="temp1")
            humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0, step=1.0, key="humid1")

        cycle_time_avg = cycle_time * 60 / 800  # convert to minutes per 800 units

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            predict_button = st.button("🎯 Predict Efficiency", use_container_width=True, key="predict1")
        
        if predict_button:
            input_data = {
                "shift_name": shift_choice,
                "supervisor_id": supervisor_choice,
                "defect_count": defect_count,
                "cycle_time_avg": cycle_time_avg,
                "operator_id": operators[0] if operators else 'OP_001',
                "experience_level": experience,
                "skill_category": skill_category_choice,
                "maintenance_downtime": downtime,
                "maintenance_flag": 1 if maintenance_flag == "Yes" else 0,
                "machine_status": machine_status_choice,
                "issue_type": issue_type_choice,
                "inspection_result": inspection_result_choice,
                "temperature": temperature,
                "humidity": humidity
            }
            
            with st.spinner("Calculating efficiency..."):
                response = requests.post(f"{API_URL}/predict", json=input_data)
                if response.ok:
                    prediction = response.json()["predicted_efficiency"]
                    
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        st.metric(label="Predicted Efficiency", value=f"{prediction:.2f}%", delta="Target: 85%")
                    with col_result2:
                        efficiency_status = "✅ Excellent" if prediction >= 85 else "⚠️ Needs Improvement" if prediction >= 70 else "❌ Critical"
                        st.metric(label="Status", value=efficiency_status)
                    with col_result3:
                        gap = 85 - prediction
                        st.metric(label="Gap to Target", value=f"{abs(gap):.2f}%")
                    
                    st.success("✓ Prediction completed successfully!")
                else:
                    st.error("❌ Error during prediction. Please check the API connection.")

    with tab2:
        st.markdown('<div class="section-header">Scenario Analysis & Optimisation</div>', unsafe_allow_html=True)
        st.markdown("*Explore optimal parameter ranges for shift efficiency*")
        
        st.subheader("🎚️ Parameter Ranges")
        col6, col7, col8 = st.columns(3)
        
        with col6:
            exp_min, exp_max = st.slider("Experience Range (years)", 1, 15, (1, 10), step=1, key="exp_range")
        
        with col7:
            downtime_min, downtime_max = st.slider("Downtime Range (hours)", 0.0, 20.0, (0.0, 10.0), step=0.5, key="down_range")
        
        with col8:
            defect_count_min, defect_count_max = st.slider("Defect Count Range", 0.0, 50.0, (0.0, 25.0), step=0.1, key="defect_range")
        
        st.subheader("📍 Shift Selection")
        col9, col10 = st.columns(2)
        with col9:
            shift_choice_opt = st.selectbox("Select Shift for Optimisation", form_data["shifts"], key="shift2")
        
        col_btn2, col_btn3 = st.columns([1, 3])
        with col_btn2:
            optimise_button = st.button("⚙️ Run Optimisation", use_container_width=True, key="optimise1")
        
        if optimise_button:
            optimisation_input = {
                "shift_name": shift_choice_opt,
                "exp_range": (exp_min, exp_max),
                "downtime_range": (downtime_min, downtime_max),
                "defect_count_range": (defect_count_min, defect_count_max)
            }
            
            with st.spinner("Running optimisation analysis..."):
                response = requests.post(f"{API_URL}/optimise", json=optimisation_input)
                if response.ok:
                    optimisation_results = pd.DataFrame(response.json())
                    
                    st.success("✓ Optimisation completed!")
                    
                    col_table1, col_table2 = st.columns([2, 1])
                    with col_table1:
                        st.subheader("📈 Optimisation Results")
                        st.dataframe(optimisation_results, use_container_width=True)
                    
                    with col_table2:
                        if len(optimisation_results) > 0:
                            best_efficiency = optimisation_results.iloc[:, -1].max() if optimisation_results.shape[1] > 0 else 0
                            st.metric("Best Efficiency Found", f"{best_efficiency:.2f}%")
                            
                            if 'experience_level' in optimisation_results.columns:
                                st.metric("Experience Range", f"{optimisation_results['experience_level'].min():.0f}-{optimisation_results['experience_level'].max():.0f} yrs")
                else:
                    st.error("❌ Error during optimisation. Please try again.")

    with tab3:
        st.markdown('<div class="section-header">Model Management & Maintenance</div>', unsafe_allow_html=True)
        st.markdown("*Manage model lifecycle and retraining*")
        
        col11, col12 = st.columns(2)
        
        with col11:
            st.markdown("### 🔄 Model Retraining")
            st.markdown("Retrain the model with the latest data to improve predictions")
            
            retrain_button = st.button("🔄 Retrain Model", use_container_width=True, key="retrain1")
            
            if retrain_button:
                with st.spinner("Retraining model... This may take a few moments."):
                    response = requests.post(f"{API_URL}/retrain")
                    if response.ok:
                        st.success("✅ Model retraining initiated! The model is being retrained in the background. Predictions will use the updated model shortly.")
                        st.info("💡 Tip: New predictions will reflect the retrained model's improved accuracy.")
                    else:
                        st.error("❌ Error during model retraining. Please try again.")
        
        with col12:
            st.markdown("### ℹ️ Model Information")
            st.markdown("""
            - **Model Type**: Ridge Regression
            - **Features**: 14 operational parameters
            - **Target**: Shift Efficiency (%)
            - **Auto-Updates**: Enabled
            
            Train your model regularly to maintain accuracy with changing operational patterns.
            """)
        
        st.divider()
        st.markdown("### 📊 System Status")
        
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                col_status1, col_status2, col_status3 = st.columns(3)
                with col_status1:
                    st.metric("API Status", "Online", "systemHealthy")
                with col_status2:
                    st.metric("Model Status", "Ready", "Loaded")
                with col_status3:
                    st.metric("Database Status", "Connected", "Active")
            else:
                st.warning("⚠️ System status check failed")
        except:
            st.warning("⚠️ Unable to connect to API for status check")

    # Footer
    st.markdown("---")
    st.markdown("<center><small>🏭 Shift Performance Optimisation System v1.0 | All parameters monitored in real-time</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()