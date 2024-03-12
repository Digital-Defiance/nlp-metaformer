"""

"""


from training import run_rust_binary, make_rust_executable, download_rust_binary
from datagen import prepare_validation_slice, write_training_slices, write_test_slices

from typing import Literal
import mlflow
from pydantic_settings import BaseSettings
from prefect import flow, task
from constants import DEV_RUST_BINARY
from env import Data, Train, Settings, Model, TrainingProcess, MLFLowSettings

@task
def log_params():
    mlflow.log_params({
        **TrainingProcess().model_dump(),
        **Model().model_dump(),
        **Train().model_dump(),
        **Data().model_dump(),
    })


@flow
def main(
    process: TrainingProcess,
    model: Model,
    train: Train,
    data: Data,
    experiment_id: int = 1,
    run_name: str | None = None,
):

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        Settings(
            process = process, 
            model = model,
            train = train,
            data = data,
            mlflow = MLFLowSettings(
                experiment_id=experiment_id,
                run_name=run_name,
                mlflow_run_id=run.info.run_id
            )
        ).to_env()
    
        log_params.submit()
        prepare_validation_slice.submit().wait()
        training_slices = write_training_slices.submit()

        path_to_rust_binary = DEV_RUST_BINARY
        if process.executable_source != DEV_RUST_BINARY:
            path_to_rust_binary = download_rust_binary(process.executable_source)
            make_rust_executable(path_to_rust_binary)

        training_loop = run_rust_binary.submit(path_to_rust_binary)

        training_slices.wait()
        write_test_slices.submit()
        training_loop.wait()


if __name__ == "__main__":

    class EnvironmentSettings(BaseSettings): 
        llmvc_environment: Literal["production", "staging", "dev"] = "production"

    main.serve(
        name = f"text-classification-{EnvironmentSettings().llmvc_environment}"
    )

