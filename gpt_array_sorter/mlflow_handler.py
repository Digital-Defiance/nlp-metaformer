import mlflow
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from contextlib import contextmanager
import os
from typing import Literal

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
            self._run_id = run.info.run_id
            yield self


    def get_status(self) -> Literal["RUNNING", "FINISHED", "FAILED"]:
        run = mlflow.get_run(self._run_id)
        return run.info.status

    def get_parameter(self, key):
        run = mlflow.get_run(self._run_id)
        return run.data.params.get(key, None)
    



