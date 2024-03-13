"""

"""


from pipelines.text_classification.run_rust import run_rust_binary, make_rust_executable, download_rust_binary
from pipelines.text_classification.data import prepare_slices
from pipelines.text_classification.constants import DEV_RUST_BINARY
from typing import Literal
import mlflow
from pydantic_settings import BaseSettings
from prefect import flow, task, get_run_logger
from pipelines.text_classification.inputs import Data, Train, Settings, Model, TrainingProcess, MLFLowSettings
from anyio import run
import duckdb
from pipelines.commons import shell_task

from numpy.random import default_rng


@task
async def log_params(settings: Settings):
    settings.to_env()
    mlflow.log_params({
        **TrainingProcess().model_dump(),
        **Model().model_dump(),
        **Train().model_dump(),
        **Data().model_dump(),
    })




@shell_task
def remove_folder(folder: str):
    return f"rm -rf {folder}"

@shell_task
def create_folder(folder: str):
    return f"mkdir -p {folder}"



@flow
async def main(
    process: TrainingProcess = TrainingProcess(),
    model: Model = Model(),
    train: Train = Train(),
    data: Data = Data(),
    experiment_id: int = 1,
    run_name: str | None = None,
): 
    # logger = get_run_logger()

    for folder in ["test", "train"]:
        await remove_folder(folder)
        await create_folder(folder)



    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:

        settings = Settings(
            process = process,
            model = model,
            train = train,
            data = data,
            mlflow = MLFLowSettings(
                experiment_id=experiment_id,
                run_name=run_name,
                mlflow_run_id=run.info.run_id
            )
        )

        settings.to_env()
        await log_params.submit(settings)

        path_to_rust_binary = DEV_RUST_BINARY
        if process.executable_source != DEV_RUST_BINARY:
            path_to_rust_binary = "./train"
            download_rust_binary(process.executable_source, path_to_rust_binary)
            make_rust_executable(path_to_rust_binary)
    

        training_loop = await run_rust_binary.submit(
            path_to_rust_binary,
            shell_env = {  key: value for key, value in Settings.from_env().yield_flattened_items() }
        )

        with duckdb.connect() as conn:
            flow_rng = default_rng(seed=42)
            prepare_slices(
                conn,
                flow_rng,
                train.epochs,
                data.slices,
                data.train_source,
                "train"
            )

        with duckdb.connect() as conn:
            flow_rng = default_rng(seed=42)
            prepare_slices(
                conn,
                flow_rng,
                1,
                data.slices,
                data.test_source,
                "test"
            )

        await training_loop.wait()



if __name__ == "__main__":
    class EnvironmentSettings(BaseSettings): 
        llmvc_environment: Literal["production", "staging", "development"] = "production"

    if EnvironmentSettings().llmvc_environment == "development":
        run(main)
    else:
        main.serve(
            name = f"text-classification-{EnvironmentSettings().llmvc_environment}"
        )

