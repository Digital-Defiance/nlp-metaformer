from celery import Celery
from celery.result import AsyncResult

celery_app = Celery(
    'celery_app',
    broker=f"redis://redis:6379/0",
    backend=f"redis://redis:6379/1",
    broker_connection_retry_on_startup=True,
    result_serializer='json',
    accept_content=['json'],
)

celery_app.conf.worker_max_memory_per_child=12000 # = 12mb, should guarantee that the worker is replaced every time after completion

def request_data(idx: int, ctx_window: int, seed: int) -> AsyncResult:
    return celery_app.send_task('prepare_data', args=[idx, ctx_window, seed])


def request_task_cleanup(task_id):
    return celery_app.send_task('cleanup_task', args=[task_id])

    

