import pandas as pd
import numpy as np
import json
import openai

# openai setting
api_key ="<Your Key Here>"
openai.api_key = api_key
import os
os.environ['OPENAI_API_KEY'] = "<Your Key Here>"
from openai import OpenAI
client = OpenAI()

def fine_tune_create(mode,size,option):
    file2 =  client.files.create(
    file=open(f"data_{mode}_{size}_1535_output50_option{option}.jsonl", "rb"),
    purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
  training_file=file2.id, 
  validation_file = None,
  model="gpt-3.5-turbo-1106",
  suffix = f"f{size}2_1535_o{option}",
  hyperparameters={
  "n_epochs": 9,
  }
)
    
if __name__ == "__main__":
    mode = 'train'
    size = '100_2'
    option='12'
    fine_tune_create(mode,size,option)