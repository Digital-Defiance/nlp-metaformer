import redis

class CustomBackend(redis.RedisBackend):
    def on_task_call(self, producer, task_id):
        pass