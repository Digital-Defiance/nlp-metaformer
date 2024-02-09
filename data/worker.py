from pydantic_settings import BaseSettings
from celery import Celery
from celery.result import AsyncResult



class Worker:
    def __init__(self) -> None:
        super().__init__()

        self.celery = Celery(
            'celery_app',
            broker=f"data.backend.CustomBackend://redis:6379/0",
            backend=f"data.backend.CustomBackend://redis:6379/1",
            broker_connection_retry_on_startup=True,
            result_serializer='pickle',

            # export C_FORCE_ROOT="true", pickle is okay here, closed network 
            accept_content=['pickle', 'json'],
        )

    def request_data(self, idx: int, ctx_window: int) -> AsyncResult:
        return self.celery.send_task('prepare_data', args=[idx, ctx_window])

