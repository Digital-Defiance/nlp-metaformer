import tiktoken
from safetensors.torch import save_file
import torch
import os
import sys
def main():
  if os.path.exists("dataset.safetensors"):
    print("dataset.safetensors file already exists")
    sys.exit(0)

  enc = tiktoken.encoding_for_model("gpt-4o")
  print("Opening files...")
  with open("pb_test.txt") as file:
    test = file.readlines()

  with open("pb_valid.txt") as file:
    val = file.readlines()

  with open("pb_train.txt") as file:
    train = file.readlines()
  
  def lines_to_tensor(lines: list[str]) -> torch.Tensor:
    print("Processing data split...")
    encoded = [
      torch.Tensor(enc.encode(line)) for line in lines
    ]
    return torch.nn.utils.rnn.pad_sequence(encoded)
  print("Processing data ")
  save_file(
    {
      "test": lines_to_tensor(test),
      "val":lines_to_tensor(val),
      "train":lines_to_tensor(train),
    }, 
    "dataset.safetensors"
  )
  print("data has been saved")


if __name__ == "__main__":
  main()
