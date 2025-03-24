import json
import openai
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import json

# openai setting
api_key =""
openai.api_key = api_key
import os
os.environ['OPENAI_API_KEY'] = ""
from openai import OpenAI
client = OpenAI()

def validate_number_format(number):
        
    if not '0.' in number:
        if not float(number)==0:
            # print(number)
            raise ValueError(f"Invalid number format: '{number}' contains spaces instead of being formatted as '0.xxx'.")
        
def get_conversation(option, size, design=0):
  if design ==0:
    test_file_path = f'../data_test_{size}_1535_output50_option{option}.jsonl'
  elif design == 1:
    test_file_path = f'/home/dl370/llm_finetune/stog_200_test_1530_output50_option7_seed1.jsonl'
  test_data = []

  with open(test_file_path, 'r') as file:
      for line in file:
          # Parse each line as a JSON object and append to the list
          json_object = json.loads(line)
          # Check if the JSON object has the key "messages"
          if "messages" in json_object:
              # print(json_object["messages"])
              test_data.append(json_object["messages"])
  print(len(test_data))
  # Initialize two lists to hold the separated data
  user_system_messages = []
  assistant_responses = []

  # Iterate over each entry in test_data
  for conversation in test_data:
      # Temporary list to hold user and system messages for the current conversation
      current_conversation = []
      for message in conversation:
          if message['role'] == 'assistant':
              # If the role is 'assistant', add the content to the assistant_responses list
              assistant_responses.append(np.array(message['content']))
          else:
              # Otherwise, add the whole message (both role and content) to the current conversation
              current_conversation.append(message)
      # Add the collected user and system messages to the user_system_messages list
      user_system_messages.append(current_conversation)
  return user_system_messages, assistant_responses

def process_res(resp,mode=10):
    # data_str = resp.tolist()

    # Removing non-numeric characters and splitting into individual elements
    # cleaned_data = data_str.strip("[]").split()

    if mode == 10:
        cleaned_data = resp.tolist().split('[')[1].split(']')[0]
        # number_list = [float(num) for num in cleaned_data.split()]
        number_strings = cleaned_data.split(' ')
        # Step 3: Convert each string in the list to a float
        number_list = []
        for num in number_strings:
            if num != '':
                validate_number_format(num.replace(',', ' '))
                # print(num)
                number_list.append(float(num.replace(',', ' ')))
    else:
        cleaned_data = resp.tolist()
        # print(cleaned_data)
        # number_list = [int(num,2) for num in cleaned_data.split()]
        number_list=[]
        for num in cleaned_data.split():
            try:
                number_list.append(int(num,2)/500)
            except ValueError:
                continue
    # print(cleaned_data)

    # Converting each element into an integer
    # number_list = [int(num)/1000 for num in cleaned_data]
    # number_list = [float(num) for num in cleaned_data]
    # number_list = [float(num) for num in cleaned_data.split()]
    return number_list

def mean_absolute_percentage_error2(true_values, predicted_values):
    """
    Calculate the Median Absolute Percentage Error (MedAPE).

    Parameters:
    - true_values (array-like): Array of true values.
    - predicted_values (array-like): Array of predicted values.
    - epsilon (float): Small value to avoid division by zero errors (default is 1e-8).

    Returns:
    - medape (float): The Median Absolute Percentage Error as a percentage.
    """
    # Convert inputs to numpy arrays for calculation
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # Calculate absolute percentage errors, adding epsilon to avoid division by zero
    absolute_percentage_errors = np.abs((true_values - predicted_values) / (true_values+1e-4))
 
    # Calculate MedAPE
    mape = np.mean(absolute_percentage_errors)

    return mape

def get_response(temperature_set,user_system_messages,test_model,design=0):
  rep_all = []
  count = 0
  for temp in temperature_set:
      responses = []

      iall=0
      for conversation in tqdm(user_system_messages):
          iall+=1
          # print(index)
          it=0
          
          while True:  # Continue trying until no ValueError
              it+=1
              response = client.chat.completions.create(
                  model=test_model,
                  messages=conversation,
                  temperature=temp
              )
              if it > 3 or design == 1:
                  responses.append(np.array(response.choices[0].message.content))
                  if design == 0:
                    print('failed in transforming', response.choices[0].message.content)
                  break
              try:
                  processed_response = process_res(np.array(response.choices[0].message.content))
                  responses.append(np.array(response.choices[0].message.content))
                  break
              except ValueError:
                  print('can not trans')
                  print(response.choices[0].message.content)
                  count += 1
                  continue
              except IndexError:
                  print('can not trans')
                  print(response.choices[0].message.content)
                  count += 1
                  continue 
      rep_all.append(responses)
      print('count',count)
  return rep_all

def get_MSE_MARE(rep_all,assistant_responses):
  modes=10

  mse_all=[]
  mae_all=[]
  for rep in rep_all:
      float_arrays= []
      float_assistant_responses = []
      for response, truth in zip(rep, assistant_responses):
          try:
              float_arrays.append(process_res(response,mode=modes))
              float_assistant_responses.append(process_res(truth,mode=modes))
          except ValueError:
              continue
          except IndexError:
              print(response)
              continue

      # Calculate MSE for each pair and average them
      mse_values = []
      mae_values=[]
      for arr1, arr2 in zip(float_arrays, float_assistant_responses):
          min_len = min(len(arr2), len(arr1))
          arr1_padded, arr2_padded = (arr1[:min_len], arr2[:min_len])
          mse = mean_squared_error(arr1_padded, arr2_padded)
          mae = mean_absolute_percentage_error2(arr1_padded, arr2_padded)
          mse_values.append(mse)
          mae_values.append(mae)
      mse_all.append(mse_values)
      mae_all.append(mae_values)
  return mse_all,mae_all

def save_res(size,option,mse_all,mae_all):
    for mse_v,mae_v in zip(mse_all,mae_all):
      avg_mse = np.median(mse_v)
      print('mse',avg_mse)
      avg_mae = np.median(mae_v)
      print('mae',avg_mae)
    # np.save(f'{size}_{option}_mse.npy',mse_all)
    # np.save(f'{size}_{option}_mae.npy',mae_all)

def main(option,size,test_model,temperature_set,i,design=0):
    user_system_messages,assistant_responses =  get_conversation(option,size,design)
    rep_all = get_response(temperature_set,user_system_messages,test_model,design)
    # np.save(f'{size}_{option}_rep.npy',rep_all)

    # save response
    with open(f'/home/dl370/llm_finetune/code/temp_res/{test_model}_{size}_{option}_{i}.pkl', 'wb') as f:
        pickle.dump(rep_all, f)
    if design == 0:
        mse_all,mae_all = get_MSE_MARE(rep_all,assistant_responses)
        print('mse',mse_all)
        save_res(size,option,mse_all,mae_all)

if __name__ == "__main__":
  option = 12
  size = 200
  test_model = ''
  print('option:',option)
  print('data size', size)
  print('model', test_model)
  temperature_set=[0, 0.5, 1]
  for i in range(1) :
    # i=i+1
    main(option,size,test_model,temperature_set,i,1)