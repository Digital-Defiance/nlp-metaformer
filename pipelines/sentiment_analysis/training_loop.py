
from prefect_shell import ShellOperation
from prefect import task
from env import Settings
from functools import wraps


def shell_task(func: callable):
    @task
    @wraps(func)
    def new_func(*args, **kwargs):
        command = func(*args, **kwargs)
        env = { key: value for key, value in Settings.from_env().yield_flattened_items() }
        ShellOperation(commands=[command], env = env).run()
    return new_func


@shell_task
def download_rust_binary(url: str) -> str:
    return f"curl -L {url} -o train"

@shell_task
def make_rust_executable(path_to_rust_binary: str) -> None:
    return f"chmod +x {path_to_rust_binary}"

@shell_task
def run_rust_binary(path_to_rust_binary: str):
    return path_to_rust_binary


