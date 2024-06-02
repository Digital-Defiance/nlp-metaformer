
from prefect_shell import ShellOperation
from prefect import task, Task

from functools import wraps
from typing import Callable
from torch import nn

def shell_task(command_factory: Callable) -> Task:
    @task(name = command_factory.__name__)
    async def prefect_task(*args, shell_env = {}, **kwargs):
        shell_command = command_factory(*args, **kwargs)
        shell_operation = ShellOperation(commands=[shell_command], env = shell_env)
        return await shell_operation.run()
    return prefect_task

import torch
from torch import nn

def make_linear(*_dim):
    dim = list(_dim)

    d0 = dim.pop(-2)
    
    d = 1
    for x in dim:
        d*=x

    return nn.Linear(d0, d, bias=False).weight.view(*_dim).contiguous()