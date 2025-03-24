import numpy as np
import pandas as pd

import torch
from util.data_class import ADMDataSet

def sample_data_points(g, s, n, seed=None):
    """
    Sample n data points from the geometry and spectrum dataset in numpy array format.
    
    Parameters:
    - g: The g array with shape [D_n, 14].
    - s: The s array with shape [D_n, 50].
    - n: The number of data points to sample.
    - seed: An optional random seed for reproducibility.
    
    Returns:
    - A tuple containing two numpy arrays: sampled g and sampled s.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Assuming g and s are aligned and we want to sample the same indices
    indices = np.random.choice(g.shape[0], size=n, replace=False)
    
    return g[indices], s[indices]

## This function takes in the datasize (log scale) to select prepared dataset in csv. format
## data_size = [100, 1000, 10000] There are three different dataset sizes for now
def load_gs_data(data_size, random_seed=None):
  ## The dataset path are saved on local server, please modify your path for reading dataset
  if data_size in (20000, 40000):
    G_train = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/g_train_150_350THz_f50_d'+str(data_size)+'.csv', header=None).values.astype('float32') ## Data type needs to be float for DNN
    s_train = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/s_train_150_350THz_f50_d'+str(data_size)+'.csv', header=None).values.astype('float32')
  else:
     G_train = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/g_train_150_350THz_f50_full.csv', header=None).values.astype('float32') ## Data type needs to be float for DNN
     s_train = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/s_train_150_350THz_f50_full.csv', header=None).values.astype('float32')
     G_train, s_train = sample_data_points(G_train, s_train, data_size, random_seed)
  
  # G_test = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/g_test_150_350THz_f50_d1000 copy.csv', header=None).values.astype('float32')
  # s_test = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/s_test_150_350THz_f50_d1000 copy.csv', header=None).values.astype('float32')

  G_test = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/g_test_150_350THz_f50_200_1.csv', header=None).values.astype('float32')
  s_test = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/s_test_150_350THz_f50_200_1.csv', header=None).values.astype('float32')

  G_val = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/g_val_150_350THz_f50_d1000.csv', header=None).values.astype('float32')
  s_val = pd.read_csv('/home/dl370/data/LLM/150_350THz_dataset/s_val_150_350THz_f50_d1000.csv', header=None).values.astype('float32')

  ## Ensure no data corruption from data loading
  assert G_train.shape[0] == s_train.shape[0], "Training set g&s sample size mismatch!"
  assert G_test.shape[0] == s_test.shape[0], "Validation set g&s sample size mismatch!"


  #print("The training set has total size of %i, geometry has %i features, and spectrum has %i frequency points"%(G_train.shape[0], G_train.shape[1], s_train.shape[1]))
  #print("The validation set has total size of %i, geometry has %i features, and spectrum has %i frequency points"%(G_test.shape[0], G_test.shape[1], s_test.shape[1]))
  # print(G_test)
  return G_train, s_train, G_test, s_test, G_val, s_val

## Helper function to normalize the input features to the range of [-1, 1]
def normalize_g(g_test):
  # g_test = np.round(g_test, decimals=3)
  for i in range(g_test.shape[1]):
    g_range = (np.max(g_test[:, i])-np.min(g_test[:, i]))/2
    g_avg = (np.max(g_test[:, i])+np.min(g_test[:, i]))/2
    g_test[:, i] = (g_test[:, i] - g_avg)/g_range  
    

  return g_test

## This function takes in the outputs from "load_gs_data" and load all numpy data into dataloader for training
def datasize_dataloader(G_train, s_train, G_test, s_test, G_val, s_val, batch_size=1):
  ## Prepare datasets as Pytorch dataloader for training
  trainset = ADMDataSet(G_train, s_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(len(G_train)//10), shuffle=True) # The batch size is set to be 1/10 of the train set. This parameter is not optimized in this project.

  testset = ADMDataSet(G_test, s_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

  valset = ADMDataSet(G_val, s_val)
  valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

  return trainloader, testloader, valloader