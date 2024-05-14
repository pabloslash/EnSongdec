import torch
import pandas as pd
import os
import json


def save_model(models_checkpoints_dir, experiment_name, model, optimizer, experiment_metadata):
    """
    Saves the model and optimizer state dictionaries to a file and appends training 
    information to a CSV file.

    If the CSV file already exists, it appends the new training information as a new row.
    If any new fields are present in the training_info dictionary, they are added as new 
    columns in the CSV file. Missing data for these new columns in previous rows are filled
    with NA values to maintain consistency.

    Args:
        models_checkpoints_dir (str): The directory path where the model and CSV files 
            will be saved.
        experiment_name (str): The base name for the files to be saved. This will be used 
            to name both the model's state dictionary file and the CSV file.
        model (torch.nn.Module): The model instance whose state dictionary will be saved.
        optimizer (torch.optim.Optimizer): The optimizer instance whose state dictionary 
            will be saved.
        experiment_metadata (dict): A dictionary containing various pieces of training 
            information to be logged in the CSV file. E.g.
            
            experiment_metadata = {
                'model_name': experiment_name,
                'dataset_filename': dataset_filename,
                'bird': bird,
                'neural_key': neural_key,
                'train_idxs': train_idxs,
                'test_idxs': test_idxs,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'layers': layers,
                'neural_history_ms': neural_history_ms,
                'num_epochs': num_epochs,
                'num_params': total_params,
                'tot_train_loss': tot_train_loss,
                'tot_val_loss': tot_val_loss,
                'tot_train_err': tot_train_err,
                'tot_val_err': tot_val_err,
                'config_path': config_path,
                'config_id': config_id,
            }

    Returns:
        None: This function does not return a value but prints out confirmation of the 
        saved files.
    """
    # Save the model and optimizer state dictionaries
    model_path = os.path.join(models_checkpoints_dir, f'{experiment_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    
    # Save the experiment_metadata dictionary to a JSON file
    metadata_path = os.path.join(models_checkpoints_dir, f'{experiment_name}_metadata.json') 
    with open(metadata_path, 'w') as output_file:
        json.dump(experiment_metadata, output_file, indent=4)
    
    print(f"Model statedict saved to {model_path}", f"\nExperiment metadata saved to {metadata_path}")

    # Path for the CSV file
    info_path = os.path.join(models_checkpoints_dir, 'models_info.csv')
    
    # Create a DataFrame for the new training information
    df_new = pd.DataFrame([experiment_metadata])

    # Check if the file exists and append if it does, otherwise write a new file
    if os.path.isfile(info_path):
        df_existing = pd.read_csv(info_path)

        # Ensure all columns exist in both DataFrames to avoid alignment issues
        for column in df_new.columns.difference(df_existing.columns):
            df_existing[column] = pd.NA  # Add missing column as NA in existing DataFrame
        for column in df_existing.columns.difference(df_new.columns):
            df_new[column] = pd.NA  # Add missing column as NA in new DataFrame
            
        idx = df_existing[df_existing['experiment_name'] == experiment_name].index
        if not idx.empty:
            df_existing.loc[idx] = df_new.values
            df_combined = df_existing
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(info_path, index=False)
    print('Training session info saved: ', info_path)


def compute_num_model_params(model):
    """
    Computes and prints the number of parameters for each layer in a model, as well as the total number of parameters.
    Args:
        model (nn.Module): The neural network model for which parameters are to be counted.
    Returns:
        int: The total number of parameters in the model.
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        print(f"Layer: {name} | Size: {parameter.size()} | Number of Parameters: {parameter.numel()}")
        total_params += parameter.numel()
    print(f"Total number of parameters in the model: {total_params}")
    return total_params
