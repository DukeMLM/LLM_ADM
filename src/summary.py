from train import LM_train, NN_train
from resultswriter import write_loss_summary_to_csv
import time
import GPUtil
import psutil
from memory_profiler import profile
import numpy as np

import csv

def log_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        return gpu.id, gpu.load * 100, gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal

def log_memory_usage():
    memory = psutil.virtual_memory()
    return memory.percent, memory.available / (1024 ** 2)

def save_results_to_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# @profile
def train_and_log(datasize, model_type, run_number,index):
    start_time = time.time()

    if model_type == 'nn':
        print(model_type + ' at ' + str(datasize), end=' ')
        median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare = NN_train(datasize, random_seed=42 * run_number, index=index)
    else:
        print(model_type + ' at ' + str(datasize), end=' ')
        median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare= LM_train(datasize, model_type, random_seed=42 * run_number,index=index)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # np.save('mmse.npy', mmse)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Log GPU and memory usage
    gpu_id, gpu_load, gpu_mem_free, gpu_mem_used, gpu_mem_total = log_gpu_usage()
    mem_usage, mem_available = log_memory_usage()
    print("GPU ",gpu_mem_used)

    # # Save results to CSV
    # result_data = [model_type, datasize, run_number + 1, elapsed_time, gpu_id, gpu_load,
    #                gpu_mem_free, gpu_mem_used, gpu_mem_total,
    #                mem_usage, mem_available, mse, mare]
    # save_results_to_csv('performance_metrics.csv', result_data)

    return median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare

if __name__ == "__main__":
    data_sizes = [1000]
    # data_sizes = [40000]
    model_types = [ 'lreg']
    # model_types = ['nn']
    mmse_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mmedare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mmare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    msmare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    smare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mse_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    medare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}

    # CSV file headers
    headers = ['Model Type', 'Data Size','time', 'median_mse', 'median_mare', 'median_smare','median_medare', 'mse', 'mare', 'smare', 'medare']

    # Initialize CSV file with headers
    filename = 'performance_metrics_913NN_cuda0.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    for datasize in data_sizes:
        for model_type in model_types:
            for i in range(1):  # Assuming 3 runs
                median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare  = train_and_log(datasize, model_type, i+3, i)

                # Append the MSE for this run to the list for the model type and datasize
                mse_results[model_type][str(datasize)].append(mse)
                mare_results[model_type][str(datasize)].append(mare)
                mmse_results[model_type][str(datasize)].append(median_mse)
                mmare_results[model_type][str(datasize)].append(median_mare)
                smare_results[model_type][str(datasize)].append(smare)
                msmare_results[model_type][str(datasize)].append(median_smare)
                mmedare_results[model_type][str(datasize)].append(median_medare)
                medare_results[model_type][str(datasize)].append(medare)
                

                # Save the result of the current run to the CSV file
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([model_type, datasize, i + 1, median_mse, median_mare, median_smare,median_medare, mse, mare, smare, medare ])
