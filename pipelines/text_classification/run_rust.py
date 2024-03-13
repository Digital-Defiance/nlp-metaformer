
from ..commons import shell_task


@shell_task
def download_rust_binary(url: str, path_to_rust_binary: str):
    return f"curl -L {url} -o {path_to_rust_binary}"

@shell_task
def make_rust_executable(path_to_rust_binary: str):
    return f"chmod +x {path_to_rust_binary}"

@shell_task
def run_rust_binary(path_to_rust_binary: str):
    shell_command = path_to_rust_binary
    return shell_command


