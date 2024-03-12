
import subprocess
from prefect_shell import ShellOperation
from prefect import get_run_logger, task
from env import Settings


class RustExitedWithError(RuntimeError):
    def __init__(self, code, error_msg):
        super().__init__(f"Command exited with non-zero code {code}: {error_msg}")

@task
def download_rust_binary(url: str) -> str:
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    save_path = "./train"
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return save_path

@task
def make_rust_executable(path_to_rust_binary: str) -> None:
    logger = get_run_logger()
    cmd = f'chmod +x {path_to_rust_binary}'
    logger.info(f"Running command: {cmd}")
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            if (output := process.stdout.readline()):
                logger.info(output.strip())

        if code != 0:
            error_msg = ""
            if process.stderr:
                error_msg = process.stderr.read()
                logger.error(error_msg)
            raise RustExitedWithError(code, error_msg)


@task
def run_rust_binary(path_to_rust_binary: str):
    ShellOperation(
        commands=[path_to_rust_binary],
        env = { key: value for key, value in Settings.from_env().yield_flattened_items() }
    ).run()