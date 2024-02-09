import celery.backends.redis   

class CustomBackend(celery.backends.redis.RedisBackend):
    def on_task_call(self, producer, task_id):
        pass

