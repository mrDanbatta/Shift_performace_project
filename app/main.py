from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import threading
import sys
import os

# Add the parent directory to Python path so src module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import load_data
from src.components.data_validation import validate
from src.pipelines.training_pipeline import run_training_pipeline
from src.pipelines.prediction_pipline import load_model
from src.logger import configure_logger
from src.exception import MyException

logger = configure_logger()

app = FastAPI(title="Shift Optimisation System API", description="API for training and evaluating the shift optimisation model.")

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://54.163.63.141:8501", # streamlit frontend
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShiftInput(BaseModel):
    shift_name: str
    supervisor_id: str
    defect_count: float
    cycle_time_avg: float
    operator_id: str
    experience_level: int
    skill_category: str
    maintenance_downtime: float
    maintenance_flag: int
    machine_status: str
    issue_type: str
    inspection_result: str
    temperature: float
    humidity: float

class Optimisation_Input(BaseModel):
    shift_name: str
    exp_range: tuple[int, int]
    downtime_range: tuple[float, float]
    defect_count_range: tuple[float, float]

@app.get("/")
def testing_api():
    return {"message": "Shift performance API is working!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/data")
def get_form_data():
    try:
        data = load_data()
        validated_data = validate(data)
        shifts = validated_data['shift_name'].unique().tolist()
        supervisors = validated_data['supervisor_id'].unique().tolist()
        skill_categories = validated_data['skill_category'].unique().tolist()
        machine_statuses = validated_data['machine_status'].unique().tolist()
        issue_types = validated_data['issue_type'].unique().tolist()
        inspection_results = validated_data['inspection_result'].unique().tolist()
        operators = validated_data['operator_id'].unique().tolist()
        return {
            "shifts": shifts,
            "supervisors": supervisors,
            "skill_categories": skill_categories,
            "machine_statuses": machine_statuses,
            "issue_types": issue_types,
            "inspection_results": inspection_results,
            "operators": operators
        }
    except MyException as e:
        logger.error(f"Error fetching form data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_shift_efficiency(input: ShiftInput):
    try:
        model = load_model()
        df = pd.DataFrame([input.dict()])
        pred = model.predict(df)[0]
        return {"predicted_efficiency": float(pred)}
    except MyException as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/optimise")
def optimise_shift(input: Optimisation_Input):
    try:
        model = load_model()
        shift_name = input.shift_name
        exp_range = input.exp_range
        downtime_range = input.downtime_range
        defect_count_range = input.defect_count_range

        exp_vals = np.arange(exp_range[0], exp_range[1]+1)
        downtime_vals = np.arange(downtime_range[0], downtime_range[1], 3)
        defect_vals = np.arange(defect_count_range[0], defect_count_range[1] +1)

        scenario = []

        for e in exp_vals:
            for d in downtime_vals:
                for dr in defect_vals:
                    scenario.append({
                        'shift_name': shift_name,
                        'supervisor_id': 'SUP_01',
                        'defect_count': dr,
                        'cycle_time_avg': 6.0 * 60 / 800,
                        'operator_id': 'OP_078',
                        'experience_level': e,
                        'skill_category': 'Senior',
                        'maintenance_downtime': d,
                        'maintenance_flag': 1 if d > 0 else 0,
                        'machine_status': 'Operational' if d == 0 else 'Under Maintenance',
                        'issue_type': 'No_Issue' if d == 0 else 'Maintenance_Issue',
                        'inspection_result': 'Pass',
                        'temperature': 22.0,
                        'humidity': 45.0
                    })

        scenario_df = pd.DataFrame(scenario)

        scenario_df['predicted_efficiency'] = model.predict(scenario_df)

        top_10 = scenario_df.sort_values(by='predicted_efficiency', ascending=False).head(10)

        return top_10.to_dict(orient='records')
    
    except MyException as e:
        logger.error(f"Error during optimisation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def retrain_pipeline():
    try:
        run_training_pipeline()
        global _cached_model
        _cached_model = None
        logger.info("Model retrained and cache cleared.")
    except MyException as e:
        logger.error(f"Error during retraining: {e}")
        raise MyException(e, sys)
    
@app.post("/retrain")
def retrain_model():
    thread = threading.Thread(target=retrain_pipeline)
    thread.start()
    return {"message": "Model retraining started in the background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
