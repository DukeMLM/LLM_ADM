import csv

def write_loss_summary_to_csv(loss, data_sizes, fname='/home/dl370/NA_LLM/res/temp.csv'):
    # Write the results to a CSV file
    with open(fname, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["ModelType"] + [str(datasize) for datasize in data_sizes]
        writer.writerow(header)
        
        # Write the rows for each model type
        for model_type, datasizes in loss.items():
            row = [model_type]
            for datasize in data_sizes:
                # Join the individual runs' MSE values into a single string for the cell
                mse_values_str = "; ".join(f"{mse:.5f}" for mse in datasizes[str(datasize)])
                row.append(mse_values_str)
            writer.writerow(row)