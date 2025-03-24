## Define the model training function here for abstractions in different model trainings
import torch
import torch.nn as nn
import numpy as np

from util.data_IO import normalize_g
from util.data_IO import load_gs_data
from util.data_IO import datasize_dataloader
import matplotlib.pyplot as plt
from memory_profiler import profile

from model import MLP, symmetric_mean_absolute_percentage_error, median_absolute_percentage_error, mean_absolute_percentage_error2
from parameters import NNParams
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import GPUtil

def log_gpu_usage():
    # Get the GPU details from GPUtil
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None, None, None, None, None

    gpu = gpus[0]  # Assuming we're only using one GPU
    gpu_id = gpu.id
    gpu_load = gpu.load * 100  # Convert to percentage
    gpu_mem_free = gpu.memoryFree
    gpu_mem_used = gpu.memoryUsed
    gpu_mem_total = gpu.memoryTotal

    return  gpu_mem_used

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
    medape = np.mean(absolute_percentage_errors)

    return medape

class time_keeper(object):
    def __init__(self, time_keeping_file="time_keeper.txt", max_running_time=9999):
        self.start = time.time()
        self.max_running_time = max_running_time * 60 * 60
        self.time_keeping_file = time_keeping_file
        self.end = -1
        self.duration = -1

    def record(self, write_number):
        """
        Record the time to the time_keeping_file, the time marked is the interval between current time and the start time
        :param write_number:
        :return:
        """
        with open(self.time_keeping_file, "a") as f:
            self.end = time.time()
            self.duration = self.end - self.start
            f.write('{},{}\n'.format(write_number, self.duration))
            if (self.duration > self.max_running_time):
                raise ValueError('Your program has run over the maximum time limit set by Ben in time_keeper function')

# def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-5):
#     # Ensure no division by zero by adding a small epsilon value
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     smape = np.mean( np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
#     return smape

# def median_absolute_percentage_error(true_values, predicted_values):
#     """
#     Calculate the Median Absolute Percentage Error (MedAPE).

#     Parameters:
#     - true_values (array-like): Array of true values.
#     - predicted_values (array-like): Array of predicted values.
#     - epsilon (float): Small value to avoid division by zero errors (default is 1e-8).

#     Returns:
#     - medape (float): The Median Absolute Percentage Error as a percentage.
#     """
#     # Convert inputs to numpy arrays for calculation
#     true_values = np.array(true_values)
#     predicted_values = np.array(predicted_values)

#     # Calculate absolute percentage errors, adding epsilon to avoid division by zero
#     absolute_percentage_errors = np.abs((true_values - predicted_values) / (true_values))

#     # Calculate MedAPE
#     medape = np.median(absolute_percentage_errors)

#     return medape

def data_preprocessing(datasize, random_seed=42):
  G_train, s_train, G_test, s_test, G_val, s_val = load_gs_data(datasize, random_seed)
  s_train = np.round(s_train, decimals=3)
  s_test = np.round(s_test, decimals=3)
  s_val = np.round(s_val, decimals=3)
  
  G_train = normalize_g(G_train)
  # print("before",G_test)
  G_test = normalize_g(G_test)
  # print(G_test)
  G_val = normalize_g(G_val)

  return G_train, s_train, G_test, s_test, G_val, s_val

def data_preprocessing_no_norm(datasize, random_seed=42):
  G_train, s_train, G_test, s_test, G_val, s_val = load_gs_data(datasize, random_seed)
  s_train = np.round(s_train, decimals=3)
  s_test = np.round(s_test, decimals=3)
  s_val = np.round(s_val, decimals=3)
  
  # G_train = normalize_g(G_train)
  # # print("before",G_test)
  # G_test = normalize_g(G_test)
  # # print(G_test)
  # G_val = normalize_g(G_val)

  return G_train, s_train, G_test, s_test, G_val, s_val

@profile
def LM_train(datasize, model_type="rf", random_seed=42, index=0):
  np.random.seed(random_seed)
  ## Load data
  G_train, s_train, G_test, s_test, G_val, s_val = data_preprocessing(datasize, random_seed)
  # print("final",G_test)

  ## Create the Regressor
  if model_type == "rf":
    model = RandomForestRegressor(n_estimators=33, min_samples_split=4, min_samples_leaf=2, max_depth=14, 
                                  random_state=random_seed)
  elif model_type == "knn":
    model = KNeighborsRegressor(n_neighbors=12)
  elif model_type == 'lreg':
    model = LinearRegression()
  else:
    TypeError("The linear model you inputed is not supported at this moment!")

  ## Train the model
  model.fit(G_train, s_train)
  # mem_usage = memory_usage((model.fit, (G_train, s_train)), interval=0.1)
  # max_mem_usage = max(mem_usage) - min(mem_usage)
  # print("Maximum memory usage during model training: {:.2f} MiB".format(max_mem_usage))

  ## Make predictions on the test set
  s_pred = model.predict(G_test)
  s_pred = np.round(s_pred, decimals=3)
  ## Calculate per-sample MSEs
  # per_sample_mare = np.array([symmetric_mean_absolute_percentage_error(s_test[i], s_pred[i]) for i in range(s_test.shape[0])])
  per_sample_mse = np.array([mean_squared_error(s_test[i], s_pred[i]) for i in range(s_test.shape[0])])
  epsilon = 1e-6
  per_sample_smare = np.array([symmetric_mean_absolute_percentage_error(s_test[i], s_pred[i]) for i in range(s_test.shape[0])])
  per_sample_medare = np.array([median_absolute_percentage_error(s_test[i], s_pred[i]) for i in range(s_test.shape[0])])
  per_sample_mare = np.array([mean_absolute_percentage_error2(s_test[i], s_pred[i]) for i in range(s_test.shape[0])])

  ## Compute the median of these MSEs
  median_mse = np.median(per_sample_mse)
  # print("Median MSE:", median_mse)
  mse = mean_squared_error(s_test, s_pred)

  # per_sample_mare = mean_absolute_percentage_error(s_test, s_pred, multioutput='raw_values')
  median_smare = np.median(per_sample_smare)
  # print("Median Absolute Relative Error:", median_smare)
  smare = np.mean(per_sample_smare)
  # print("Mean Absolute Relative Error:", smare)

  median_mare = np.median(per_sample_mare)
  print("Median Absolute Relative Error:", median_mare)
  mare = mean_absolute_percentage_error(s_test, s_pred)
  print("Mean Absolute Relative Error:", mare)
  mare = np.mean(per_sample_mare)
  print("Mean Absolute Relative Error:", mare)
  np.save(f'pred_{model_type}_{datasize}_{index}.npy', s_pred)
  np.save(f'truth_{model_type}_{datasize}_{index}.npy', s_test)


  median_medare = np.median(per_sample_medare)
  # print("Median Absolute Relative Error:", median_medare)
  medare = np.mean(per_sample_medare)
  # print("Mean Absolute Relative Error:", medare)
  
  return median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare


def NN_train(datasize, random_seed=42, index=0):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)

  ## Loading data into numpy/dataloader
  G_train, s_train, G_test, s_test, G_val, s_val = data_preprocessing(datasize, random_seed)
  trainloader, testloader, valloader = datasize_dataloader(G_train, s_train, G_test, s_test, G_val, s_val)

  ## Initialize the parameters for NN
  params = NNParams(learning_rate=1e-3, num_layer=10, num_neuron=2000)
  
  ## Model initialization with loss, optimizer, and scheduler
  model = MLP(params.num_layer, params.num_neuron, G_train.shape[1], s_train.shape[1])
  model = model.to(device)
  Loss = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=params.lr_decay_rate, 
                                                            patience=10, threshold=1e-4)
  
  ## Training section
  train_loss_arr, val_loss_arr = model.train_model(device, trainloader, valloader, Loss, optimizer, scheduler=scheduler, 
                                             num_epochs=params.num_epoch, eval_step=params.eval_step)
  
  ## Separate testset
  model.load_state_dict(torch.load('model-best.pth'))
  index = f"{index}_{datasize}"
  mse_val_arr, smare_val_arr, medare_val_arr, mare_val_arr = model.validate_model(device, testloader, Loss, index)
  
  # print(np.mean(mse_val_arr))
  # print(np.mean(mae_val_arr))
  # print(np.mean(mare_val_arr))

  return np.median(mse_val_arr), np.median(mare_val_arr), np.median(smare_val_arr), np.median(medare_val_arr), np.mean(mse_val_arr),np.mean(mare_val_arr), np.mean(smare_val_arr), np.mean(medare_val_arr)

def NN_test(datasize, random_seed=42):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)

  ## Loading data into numpy/dataloader
  G_train, s_train, G_test, s_test, G_val, s_val = data_preprocessing(datasize, random_seed)
  trainloader, testloader, valloader = datasize_dataloader(G_train, s_train, G_test, s_test, G_val, s_val)

  ## Initialize the parameters for NN
  params = NNParams(learning_rate=1e-3, num_layer=12, num_neuron=2000)
  
  ## Model initialization with loss, optimizer, and scheduler
  model = MLP(params.num_layer, params.num_neuron, G_train.shape[1], s_train.shape[1])
  model = model.to(device)
  Loss = nn.MSELoss()
  ## Separate testset
  model.load_state_dict(torch.load('/home/dl370/model/model-best.pth'))
  mse_val_arr, mae_val_arr, mare_val_arr = model.validate_model(device, testloader, Loss)
  print(np.mean(mse_val_arr))
  print(np.mean(mae_val_arr))
  print(np.mean(mare_val_arr))

  return np.mean(mse_val_arr), np.mean(mae_val_arr), np.mean(mare_val_arr)