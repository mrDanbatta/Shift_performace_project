import mlflow
from mlflow.tracking import MlflowClient
import sys
import mlflow.sklearn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.logger import configure_logger
from src.exception import MyException

class ModelPusher:
    def __init__(self):
        self.logger = configure_logger()
        
        # Get DagsHub token from environment
        dagshub_token = os.getenv('DAGSHUB_USER_ACCESS_TOKEN')
        
        try:
            if dagshub_token:
                # Configure MLflow for DagsHub using proper authentication
                repo_owner = 'mrDanbatta'
                repo_name = 'Shift_optimisation_system'
                
                # Set the tracking and registry URI (without token in URL)
                uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
                mlflow.set_tracking_uri(uri)
                mlflow.set_registry_uri(uri)
                
                # Set authentication credentials via environment variables
                # DagsHub uses HTTP Basic Auth with token as password
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                
                self.logger.info("MLflow tracking configured with DagsHub remote backend.")
            else:
                self.logger.info(
                    "DAGSHUB_USER_ACCESS_TOKEN not found. Using local MLflow backend. "
                    "Models will be tracked locally in mlruns/ directory."
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to configure DagsHub tracking: {e}. "
                "Using local MLflow backend as fallback."
            )
        
        self.client = MlflowClient()
        self.registered_model_name = "ShiftPerformanceModel"

    def get_best_exsisting_model_metrics(self):
        """
        Retrieves the best existing model from the MLFlow registry based on the lowest MAE.
        Returns a tuple: (r2_score, mae_score) of the best model.
        """
        try:
            try:
                versions = self.client.get_latest_versions(self.registered_model_name)
            except Exception:
                # Model doesn't exist yet
                self.logger.info("No existing model found in the registry.")
                return None, None
                
            if not versions:
                self.logger.info("No existing model found in the registry.")
                return None, None
            
            best_r2, best_mae = None, None
            for v in versions:
                run_id = v.run_id
                run = self.client.get_run(run_id)
                r2 = run.data.metrics.get("r2_score")
                mae = run.data.metrics.get("mae_score")

                if r2 is None or mae is None:
                    continue

                # Pick the best model based on the lowest MAE and highest R2
                if (best_r2 is None) or (r2 > best_r2) or (r2 == best_r2 and mae < best_mae):
                    best_r2 = r2
                    best_mae = mae

            return best_r2, best_mae
        except Exception as e:
            self.logger.error(f"Error retrieving existing model: {e}")
            return None, None
        
    def push_model(self, model, r2_score: float, mae_score: float) -> bool:
        """
        Pushes the new model to the MLFlow registry if it performs better than the existing model."""

        try:
            self.logger.info("Comparing new model with existing model in the registry.")
            old_r2, old_mae = self.get_best_exsisting_model_metrics()

            push_new_model = False
            if old_r2 is None:
                self.logger.info("No existing model to compare against. Pushing new model.")
                push_new_model = True
            elif r2_score > old_r2:
                self.logger.info(f"New model has better R2 score ({r2_score}) than existing model ({old_r2}). Pushing new model.")
                push_new_model = True
            elif r2_score == old_r2:
                if mae_score < old_mae:
                    push_new_model = True
                    self.logger.info(f"New model has same R2 score but better MAE score ({mae_score}) than existing model ({old_mae}). Pushing new model.")
                else:
                    self.logger.info(f"New model has same R2 score and worse MAE score ({mae_score}) than existing model ({old_mae}). Skipping push.")
            else:
                self.logger.info(f"New model has worse R2 score ({r2_score}) than existing model ({old_r2}). Skipping push.")

            if not push_new_model:
                return False
            
            # log new model to MLFlow and register it
            try:
                with mlflow.start_run():
                    mlflow.log_metric("r2_score", r2_score)
                    mlflow.log_metric("mae_score", mae_score)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=self.registered_model_name
                    )
                    self.logger.info("New model logged and registered successfully.")
                    return True
            except Exception as mlflow_error:
                # Check if it's an authentication error
                error_str = str(mlflow_error).lower()
                if "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    self.logger.error(
                        f"Authentication failed with MLflow/DagsHub: {mlflow_error}. "
                        "Please verify DAGSHUB_USER_ACCESS_TOKEN is correct and the repository exists."
                    )
                else:
                    self.logger.error(f"Error pushing model to registry: {mlflow_error}")
                raise MyException(mlflow_error, sys)
            
        except Exception as e:
            self.logger.error(f"Error in push_model: {e}")
            raise MyException(e, sys)

