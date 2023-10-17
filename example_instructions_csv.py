# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import pandas as pd
import json
import math

from llama import Llama

instruction_template = f""

# Function to create the list of elements
def create_instruction(row):
    # system_message = f""" """  
    instruction = f"""
        Architecture:
        {row['architecture']}

        Code:
        {row['code']}

        Performance:
        {row['performance']}

        {row['task']}
    """
    return [{
        "id": row["id"],
        "role": "user",
        "content": instruction
    }]

def main(
    csv_path: str,
    out_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # instructions = [
    #     [
    #         {
    #             "role": "user",
    #             "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
    #         }
    #     ],
    #     [
    #         {
    #             "role": "user",
    #             "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
    #         }
    #     ],
    #     [
    #         {
    #             "role": "system",
    #             "content": "Provide answers in JavaScript",
    #         },
    #         {
    #             "role": "user",
    #             "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
    #         }
    #     ],
    # ]
    
    # read & build instructions from csv file or parquet file
    df = pd.read_csv(csv_path)

    # Apply the function to create the list of elements
    df['elements'] = df.apply(create_instruction, axis=1)

    # Extract the list of elements
    instructions = df['elements'].tolist()
    print(json.dumps(instructions, indent=4))


    
    out = []
    for batch in range(0, math.ceil(len(instructions) / max_batch_size)):
      start = batch * max_batch_size
      batch_size = len(instructions) % max_batch_size if (batch * (max_batch_size+1)) > len(instructions) else max_batch_size
      results = generator.chat_completion(
          instructions[start:batch_size],   # type: ignore
          max_gen_len=max_gen_len,
          temperature=temperature,
          top_p=top_p,
      )

      for instruction, result in zip(instructions, results):
          for msg in instruction:
              print(f"{msg['role'].capitalize()}: {msg['content']}\n")
          print(
              f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
          )
          out.append({"id": instruction[0]['id'], "result": result['generation']['content']})
              
          print("\n==================================\n")
    
    data = pd.DataFrame(out)
    data.to_csv(out_path, index=False)

if __name__ == "__main__":
    fire.Fire(main)
