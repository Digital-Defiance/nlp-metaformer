

import mlflow


class MyBaseSettingsMixin:

    def save_to_mlflow(self):
        for key, value in self.dict().items():
            mlflow.log_param(key, value)

    @classmethod
    def load_from_mlflow(cls):
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        return cls(**run.data.params)
    
    def to_exports(self) -> dict:
        env_prefix = self.Config.env_prefix
        return {
            env_prefix + key: str(value)
            for key, value in self.dict().items()
        }

