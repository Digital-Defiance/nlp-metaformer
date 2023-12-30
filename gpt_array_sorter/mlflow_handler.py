import mlflow
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from contextlib import contextmanager
import os

load_dotenv()

class MLFlowHandler(BaseSettings):
    EXPERIMENT_ID: int
    TRACKING_URL: str
    LOG_SYSTEM_METRICS: bool = True

    _run_id: str = None

    @classmethod
    @contextmanager
    def start_run(cls):
        self = cls()
        mlflow.set_tracking_uri(self.TRACKING_URL)

        with mlflow.start_run(experiment_id=self.EXPERIMENT_ID) as run:
            mlflow.log_param("status", "STARTING")
            self._run_id = run.info.run_id
            yield self


    @classmethod
    @contextmanager
    def continue_run(cls):
        self = cls()
        mlflow.set_tracking_uri(self.TRACKING_URL)
        mlflow.enable_system_metrics_logging()
        with mlflow.start_run(
            run_id=os.environ.get("RUN_ID"),
            log_system_metrics=self.LOG_SYSTEM_METRICS,
        ) as run:
            mlflow.log_param("status", "RESUMING")
            self._run_id = run.info.run_id
            yield run
                

    def is_active(self):
        run = mlflow.get_run(self._run_id)
        status = run.data.params['status']

        if status == 'FINISHED':
            return False
        
        return True


