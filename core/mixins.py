

import mlflow
from typing import Self


class MLFlowSaveAndLoadMixin:

    def save_to_mlflow(self):
        for key, value in self.dict().items():
            mlflow.log_param(key, value)

    @classmethod
    def load_from_mlflow(cls) -> Self:
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        return cls(**run.data.params)

