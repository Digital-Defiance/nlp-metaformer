
from worker import Worker

worker = Worker()
print("Worker has been initialized.")


task = worker.request_data(0, 100)
print("Request for slice 0 has been sent.")
print("Task id is ", task)

result = task.get()

print("Task has been completed.")

print(result)
print(result[0].shape)
print(result[1].shape)
task.forget()



