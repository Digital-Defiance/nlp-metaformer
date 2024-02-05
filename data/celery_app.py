from celery import Celery



celery_app = Celery('celery_app', broker='redis://localhost:6379/0')

