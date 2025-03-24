from train import LM_train, NN_train
from resultswriter import write_loss_summary_to_csv

if __name__ == "__main__":
    data_sizes = [100, 1000, 10000, 20000, 40000]
    model_types = ['lreg', 'knn', 'rf', 'nn']
    mse_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mae_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}
    mare_results = {model_type: {str(datasize): [] for datasize in data_sizes} for model_type in model_types}

    for datasize in data_sizes:
        for model_type in model_types:
            for i in range(3):  # Assuming 3 runs
                if model_type == 'nn':
                    print(model_type+' at '+str(datasize), end=' ')
                    mse, mae, mare = NN_train(datasize, random_seed=42*i)
                else:
                    print(model_type+' at '+str(datasize), end=' ')
                    mse, mae, mare = LM_train(datasize, model_type, random_seed=42*i)
                # Append the MSE for this run to the list for the model type and datasize
                mse_results[model_type][str(datasize)].append(mse)
                mae_results[model_type][str(datasize)].append(mae)
                mare_results[model_type][str(datasize)].append(mare)
    
    write_loss_summary_to_csv(mse_results, data_sizes, 
                              fname='/hpc/group/tarokhlab/yd105/LLM_ADM/mse_summary_052624.csv')
    
    write_loss_summary_to_csv(mae_results, data_sizes, 
                              fname='/hpc/group/tarokhlab/yd105/LLM_ADM/mae_summary_052624.csv')
    
    write_loss_summary_to_csv(mare_results, data_sizes, 
                              fname='/hpc/group/tarokhlab/yd105/LLM_ADM/mare_summary_052624.csv')
