
import tiktoken
import numpy as np

input_file_path = "raw_data.txt"

with open(input_file_path, 'r') as file:
    data: str = file.read()

size_of_text = len(data)
thresh_size = int(size_of_text * 0.9)
train_data, val_data = data[:thresh_size], data[thresh_size:]
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
np.save("train.bin", train_ids)
np.save("val.bin", val_ids)
print(enc.max_token_value)



