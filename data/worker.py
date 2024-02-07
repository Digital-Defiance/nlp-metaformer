from pydantic_settings import BaseSettings
from celery import Celery
from celery.result import AsyncResult


class Worker(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    celery: Celery | None = None

    class Config:
        env_prefix = "REDIS_"

    def __init__(self) -> None:
        super().__init__()

        def get_redis_uri(db = 0):
            return f"redis://{self.host}:{self.port}/{db}"

        self.celery = Celery(
            'celery_app',
            broker=get_redis_uri(db=1),
            backend=get_redis_uri(db=2),
            broker_connection_retry_on_startup=True,
            result_serializer='pickle',

            # export C_FORCE_ROOT="true", pickle is okay here, closed network 
            accept_content=['pickle', 'json'],
        )

    def request_data(self, idx: int, ctx_window: int) -> AsyncResult:
        return self.celery.send_task('prepare_data', args=[idx, ctx_window])

