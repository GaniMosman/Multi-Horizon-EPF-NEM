import os
import sys
import time
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from Modules import Models
from Modules import Utils
from Modules.Trainer import Trainer
from Modules.DataPreparation import DataPreparation as DataPreparation


def run_HPO_pipeline(
                dataset, 
                project_root,
                features_dim = 'U',
                dataset_id = "QLD",
                target_col = 0, 
                train_ratio = 0.8,
                cutoff_date = None, 
                granularity = "30min",
                scale_method = "MinMax",
                temp_f = False,
                input_seq_len = 336,
                output_seq_len = 48,
                batch_size = 128,
                model_id = 'LSTM',
                search_method = "Grid",
                search_space = None,
                loss_function = None,
                optimizer = None,
                epochs = None,
                device = "cuda", 
                verbose = True
                ):

    print(f"***** Shape of the dataset: {dataset.shape} *****\n")

    # If the experiment is with Univariate input
    if features_dim in ('U', 'u'):
        dataset = dataset.iloc[:, [target_col]]
    elif features_dim in ('M', 'm'):
        pass
    else: 
        print("❌ features_dim must be one of 'U', 'u', 'M', or 'm'!\n")
        sys.exit(1)

    # Init data preparation class
    data_prep = DataPreparation()
    
    print("┌---------------------------------------------┐")
    print("|          Data Preparation Initiated!        |")
    print("└---------------------------------------------┘\n\n")
    
    # Downsampling
    if granularity != "5min":
        dataset = data_prep.downsample_data(dataset, freq = granularity)
        print(f"✅ Dataset Downsampled to {granularity}!\n")
        print(f" Shape of the dataset after downsampling: {dataset.shape}\n")

    if features_dim in ('U', 'u'):
        print(" ----- First few rows -----")
        print(dataset.head(5))
        print()
        print(" ----- Last few rows -----")
        print(dataset.tail(5))
        print()
    
    # Train, validation, test split
    train_data, val_data, _ = data_prep.split_time_series(dataset, train_ratio=train_ratio, cutoff_date=cutoff_date)
    print("✅ Dataset Splitted into Train, Validation, and Test Sets!\n")
    print(f" Shape of the Training set: {train_data.shape}")
    print(f" Shape of the Validation set: {val_data.shape}\n")

    if features_dim in ('U', 'u'):
         # Scale the data
        train_scaled, val_scaled, _ = data_prep.scale(train_data, val_data, method=scale_method, target_col=0)
        print(f"✅ Scaled Using {scale_method} Method!\n")

        # Create sequences
        X_train, y_train, _, _ = data_prep.create_sequences(train_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=0)
        X_val, y_val, _, _ = data_prep.create_sequences(val_scaled.values, input_length=input_seq_len,
                                                    output_length=output_seq_len, target_col=0)

    else:
        # Scale the data
        train_scaled, val_scaled, _ = data_prep.scale(train_data, val_data, method=scale_method, target_col=target_col)
        print(f"✅ Scaled Using {scale_method} Method!\n")

        # Create sequences
        X_train, y_train, _, _  = data_prep.create_sequences(train_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=target_col)
        X_val, y_val, _, _  = data_prep.create_sequences(val_scaled.values, input_length=input_seq_len,
                                                    output_length=output_seq_len, target_col=target_col)

    print(f"✅ Sequences Created with input length {input_seq_len}, and output length {output_seq_len}!\n")
    print(f" Shape of Training sequences: X--> {X_train.shape}, y--> {y_train.shape}")
    print(f" Shape of Validation sequences: X--> {X_val.shape},  y--> {y_val.shape}\n")


    if temp_f:
        # Temp features generation
        train_temp = data_prep.extract_temporal_features(train_data.index)
        val_temp = data_prep.extract_temporal_features(val_data.index)
        print("✅ Temporal features are created!\n")

        # Scale temp features
        train_temp_scaled, val_temp_scaled, _ = data_prep.scale(train_temp, val_temp, method=scale_method)
        print(f"✅ Temporal features are scaled using {scale_method}!\n")

        # Temp features sequences creation
        train_temp_seq, _ = data_prep.create_sequences(train_temp_scaled.values, input_length=input_seq_len, output_length=output_seq_len)
        val_temp_seq, _ = data_prep.create_sequences(val_temp_scaled.values, input_length=input_seq_len, output_length=output_seq_len)
        print("✅ Temporal features sequences are created!\n")

        print(f" Shape of Training sequences (Temporal Features): {train_temp_seq.shape}")
        print(f" Shape of Validation sequences (Temporal Features): {val_temp_seq.shape}\n")

        # Data loaders with temporal features sequences
        train_loader = data_prep.create_dataloader(X_train, train_temp_seq, y_train, batch_size=batch_size, shuffle=True)
        val_loader = data_prep.create_dataloader(X_val, val_temp_seq, y_val, batch_size=batch_size, shuffle=False)
        
    else: 
        # Data loaders without temporal features sequences
        train_loader = data_prep.create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = data_prep.create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
        
    print(f"✅ Data Loaders created with batch size {batch_size}!\n\n")

    print("┌---------------------------------------------┐")
    print("|          Data Preparation Finished!         |")
    print("└---------------------------------------------┘\n\n")

    
    # Get the model_class and manadatory arguments
    model_class, min_args = Utils.get_model_class_and_args(model_id, X_train, y_train, Models)

    # Printing
    box_width = 50
    title = f"Hyperparamters search for {model_id} model! "
    padding_length = (box_width - len(title)) // 2
    padding_left = " " * padding_length
    padding_right = " " * (box_width - len(title) - padding_length) 
    print("\n┌" + "-" * box_width + "┐")
    print(f"│{padding_left}{title}{padding_right}│")
    print("└" + "-" * box_width + "┘\n\n")
    

    # Default values if not passed
    loss_function = loss_function or nn.L1Loss()
    epochs = epochs or 30
    
    # HPO function
    keys, values = zip(*search_space.items())
    total_combos = np.prod([len(v) for v in values])

    if verbose:
        combo_iterator = itertools.product(*values)
    else:
        combo_iterator = tqdm(itertools.product(*values), total=total_combos, desc="Grid Search")

    results = []

    print("\n✅ --------------- Search Started! --------------- ✅\n")
    print(f"🔍 Total hyperparameter combinations to search: {total_combos}!\n")

    search_start_time = time.time()

    for i, combo in enumerate(combo_iterator):
        Utils.set_seed(42)
        parameters = dict(zip(keys, combo))
        model_instance_kwargs = {**min_args, **parameters}
        lr = model_instance_kwargs.pop("lr", 1e-3)
        model = model_class(**model_instance_kwargs).to(device)

        if optimizer is None:
            optim_instance = torch.optim.Adam(model.parameters(), lr=lr)
        elif isinstance(optimizer, type):
            optim_instance = optimizer(model.parameters(), lr=lr)
        else:
            optim_instance = optimizer
            
         # Learning rate scheduler
        lr_scheduler = torch. optim.lr_scheduler.ReduceLROnPlateau(
                                                        optim_instance,
                                                        mode="min",       
                                                        factor=0.5,      
                                                        patience=3,
                                                        cooldown=1,
                                                        min_lr=1e-5
                                                        )

        # Init Trainer
        trainer = Trainer()

        _, val_losses = trainer.train(
                                    model = model, 
                                    train_data = train_loader, 
                                    val_data = val_loader, 
                                    epochs = epochs,
                                    loss_function = loss_function,
                                    optimizer = optim_instance,
                                    device = device,
                                    early_stopping_patience = 10, 
                                    min_delta = 0, #1e-4,
                                    lr_scheduler = lr_scheduler,
                                    verbose = False
                                    )
        print()
        
        avg_val_loss= sum(val_losses) / len(val_losses)

        if verbose:
            print(f"{i+1}/{total_combos} {parameters}, Average validation loss: {avg_val_loss:.6f}, Best validation loss: {trainer.best_val_loss_overall:.6f}\n\n")
                              
        results.append({"trial_num": i, **parameters, "avg_val_loss": avg_val_loss, 
                        "best_val_loss": trainer.best_val_loss_overall,
                        "best_epoch": trainer.best_epoch, "total_epoch": trainer.total_epoch})
    
    search_end_time = time.time()
    total_time = search_end_time - search_start_time

    # Convert the results to dataframe
    results_df = pd.DataFrame(results)

    # Return best params and loss
    best_params, best_loss = Utils.return_best_params(results_df)

    # Save the log to the csv
    log_dir = os.path.join(project_root, "Outputs/HPO/Logs")
    results_file_name = f"{model_id}_{dataset_id}_{features_dim}_{granularity}_{input_seq_len}_{output_seq_len}.csv"
    results_df.to_csv(os.path.join(log_dir, results_file_name), index=False)


    # Save the best parameters
    best_params_path = os.path.join(project_root, "Outputs/HPO/Optimal")
    best_params = {k: v for k, v in best_params.items() if k in search_space}
    model_key = f"{model_id}_{dataset_id}_{features_dim}_{granularity}_{input_seq_len}_{output_seq_len}"
    Utils.save_best_params(best_params_path, model_key, best_params)

    print(f"\n===== Total time taken:{total_time:.4f} Sec =====\n")
    print("✅ --------------- Search Finished! -------------------- ✅\n\n")

    # Print Results
    print("************************  Results  ************************\n")
    print(" Best params:", best_params)
    print(f" Best validation loss: {best_loss:.6f}")
    print("\n***********************************************************")