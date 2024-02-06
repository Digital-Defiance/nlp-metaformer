
from worker import Worker

worker = Worker()
task = worker.request_data(0)
result = task.get()
    
print(result)
print(result[0].shape)
print(result[1].shape)



