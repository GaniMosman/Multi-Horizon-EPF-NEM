import os
import sys
import time
import torch
import torch.nn as nn
from Modules import Models
from Modules import Utils
from Modules.Trainer import Trainer
from Modules.Evaluation import Evaluation 
from Modules.DataPreparation import DataPreparation as DataPreparation

def run_pipeline(
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
                model_args = None,
                loss_function = None,
                optimizer = None,
                lr = None,
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
        print( " ----- Last few rows -----")
        print(dataset.tail(5))
        print()
    
    # Train, validation, test split
    train_data, val_data, test_data = data_prep.split_time_series(dataset, train_ratio=train_ratio, cutoff_date=cutoff_date)
    print("✅ Dataset Splitted into Train, Validation, and Test Sets!\n")
    print(f" Shape of the Training set: {train_data.shape}")
    print(f" Shape of the Validation set: {val_data.shape}")
    print(f" Shape of the Test set: {test_data.shape}\n")

    # Save splitted datasets
    split_path = os.path.join(project_root, "Outputs/Dataset/Splits")
    Utils.save_splits(split_path, train_data, val_data, test_data, name = f"{dataset_id}_{features_dim}_{granularity}")
    
    if features_dim in ('U', 'u'):
         # Scale the data
        train_scaled, val_scaled, test_scaled = data_prep.scale(train_data, val_data, test_data, method=scale_method, target_col=0)
        print(f"✅ Scaled Using {scale_method} Method!\n")

        # Create sequences
        X_train, y_train, X_train_time, y_train_time = data_prep.create_sequences(train_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=0, 
                                                    index = train_scaled.index)
        X_val, y_val, X_val_time, y_val_time = data_prep.create_sequences(val_scaled.values, input_length=input_seq_len,
                                                    output_length=output_seq_len, target_col=0, 
                                                    index = val_scaled.index)
        X_test, y_test, X_test_time, y_test_time = data_prep.create_sequences(test_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=0, 
                                                    index = test_scaled.index)
    else:
        # Scale the data
        train_scaled, val_scaled, test_scaled = data_prep.scale(train_data, val_data, test_data, method=scale_method, target_col=target_col)
        print(f"✅ Scaled Using {scale_method} Method!\n")

        # Create sequences
        X_train, y_train, X_train_time, y_train_time  = data_prep.create_sequences(train_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=target_col, 
                                                    index = train_scaled.index)
        X_val, y_val, X_val_time, y_val_time  = data_prep.create_sequences(val_scaled.values, input_length=input_seq_len,
                                                    output_length=output_seq_len, target_col=target_col, 
                                                    index = val_scaled.index)
        X_test, y_test, X_test_time, y_test_time = data_prep.create_sequences(test_scaled.values, input_length=input_seq_len, 
                                                    output_length=output_seq_len, target_col=target_col, 
                                                    index = test_scaled.index)

    print(f"✅ Sequences Created with input length {input_seq_len}, and output length {output_seq_len}!\n")
    print(f" Shape of Training sequences: X--> {X_train.shape}, y--> {y_train.shape}")
    print(f" Shape of Validation sequences: X--> {X_val.shape}, y--> {y_val.shape}")
    print(f" Shape of Test sequences: X--> {X_test.shape}, y--> {y_test.shape}\n")

    
    # Save scaler
    '''
    scaler_path = os.path.join(project_root, "Outputs/Dataset/Scalers")
    Utils.save_scaler(data_prep.scaler, scaler_path, filename = f"{scale_method}_{dataset_id}_{features_dim}_{granularity}")
    '''
    # Save the test sequences with the timestamp information
    '''
    sequence_path = os.path.join(project_root, "Outputs/Dataset/Sequences")
    Utils.save_sequences(sequence_path, X_test, y_test, X_test_time, y_test_time, 
                    name = f"test_sequences_{dataset_id}_{scale_method}_{features_dim}_{granularity}_{input_seq_len}_{output_seq_len}")
    '''

    if temp_f:
        # Temp features generation
        train_temp = data_prep.extract_temporal_features(train_data.index)
        val_temp = data_prep.extract_temporal_features(val_data.index)
        test_temp = data_prep.extract_temporal_features(test_data.index)
        print("\n✅ Temporal features created!\n")

        # Scale temp features
        train_temp_scaled, val_temp_scaled, test_temp_scaled = data_prep.scale(train_temp, val_temp, test_temp, method=scale_method)
        print(f"✅ Temporal features are scaled using {scale_method}!\n")

        # Temp features sequences creation
        train_temp_seq, _ = data_prep.create_sequences(train_temp_scaled.values, input_length=input_seq_len, output_length=output_seq_len)
        val_temp_seq, _ = data_prep.create_sequences(val_temp_scaled.values, input_length=input_seq_len, output_length=output_seq_len)
        test_temp_seq, _ = data_prep.create_sequences(test_temp_scaled.values, input_length=input_seq_len, output_length=output_seq_len)
        print("✅ Temporal features sequences are created!\n")

        print(f" Shape of Training sequences (Temporal Features): {train_temp_seq.shape}")
        print(f" Shape of Validation sequences (Temporal Features): {val_temp_seq.shape}")
        print(f" Shape of Test sequences (Temporal Features): {test_temp_seq.shape}\n")
        
        ''' 
        # Save the test temp sequences
        Utils.save_temp_sequences(sequence_path, test_temp_seq,
                    name = f"test_sequences_{dataset_id}_{scale_method}_{features_dim}_{granularity}_{input_seq_len}_{output_seq_len}")
        '''

        # Data loaders with temporal features sequences
        train_loader = data_prep.create_dataloader(X_train, train_temp_seq, y_train, batch_size=batch_size, shuffle=True)
        val_loader = data_prep.create_dataloader(X_val, val_temp_seq, y_val, batch_size=batch_size, shuffle=False)
        test_loader = data_prep.create_dataloader(X_test, test_temp_seq, y_test, batch_size=batch_size, shuffle=False)
        
    else: 
        # Data loaders without temporal features sequences
        train_loader = data_prep.create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = data_prep.create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
        test_loader = data_prep.create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    print(f"\n✅ Data Loaders Created with batch size {batch_size}!\n\n")

    print("┌---------------------------------------------┐")
    print("|          Data Preparation Finished!         |")
    print("└---------------------------------------------┘\n\n")

    # Load model with all the necessary arguments
    best_params_path = os.path.join(project_root, "Outputs/HPO/Optimal")
    model_key = f"{model_id}_{dataset_id}_{features_dim}_{granularity}_{input_seq_len}_{output_seq_len}"
   
    model_class, min_args = Utils.get_model_class_and_args(model_id, X_train, y_train, Models)

    if model_args is None:
        model_args = Utils.load_best_params(best_params_path, model_key)
    else:
        print(f"✅ Using provided args for {model_id}!\n")

    model_args = {**min_args, **model_args}
    model_args_WLR = {k: v for k, v in model_args.items() if k != "lr"}

    fixed_seeds = [0, 42, 84, 168, 336]
    for seed in fixed_seeds:
        print(f"\n✅ ---------------------- Running with seed {seed} ---------------------- ✅\n\n")
        Utils.set_seed(seed)
        model = model_class(**model_args_WLR).to(device)

        # Default values if not passed
        loss_func = loss_function or nn.L1Loss()
        lr_run = lr if lr is not None else model_args.get("lr", 0.01)
        print(f" Using Loss Function: {loss_func}")
        epochs_run = epochs or 30
        
        if optimizer is None:
            optim_instance = torch.optim.Adam(model.parameters(), lr=lr_run)
        elif isinstance(optimizer, type):
            optim_instance = optimizer(model.parameters(), lr=lr_run)
        else:
            optim_instance = optimizer

        print(f" Using Optimizer: {optim_instance.__class__.__name__}\n")
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_instance,
            mode="min",       
            factor=0.5,      
            patience=3,
            cooldown=1,
            min_lr=1e-5
            )

        # Printing
        box_width = 45
        title = f" {model_id} Model Initiated! "
        padding_length = (box_width - len(title)) // 2
        padding_left = " " * padding_length
        padding_right = " " * (box_width - len(title) - padding_length) 
        print("\n┌" + "-" * box_width + "┐")
        print(f"│{padding_left}{title}{padding_right}│")
        print("└" + "-" * box_width + "┘\n\n")

        # Init Trainer
        trainer = Trainer()

        path = f"{model_id}_{dataset_id}_{features_dim}_{granularity}_{scale_method}_{str(loss_function)[:-2]}_{input_seq_len}_{output_seq_len}_{seed}"
        save_best_model_path = os.path.join(project_root, f"Outputs/Models/{path}.pth")
        log_csv_path = os.path.join(project_root, f"Outputs/Logs/{path}.csv")

        # Train and Validation
        print(" ✅ ----- Training and Validation Started! ----- ✅\n")

        training_start_time = time.time()
        train_losses, val_losses = trainer.train(
                                            model = model, 
                                            train_data = train_loader, 
                                            val_data = val_loader, 
                                            epochs = epochs_run,
                                            loss_function = loss_func,
                                            optimizer = optim_instance,
                                            device = device,
                                            early_stopping_patience = 10, 
                                            min_delta = 0, #1e-4,
                                            lr_scheduler = lr_scheduler,
                                            save_best_model_path = save_best_model_path,
                                            log_csv_path = log_csv_path,
                                            verbose = verbose
                                            )

        print(" ✅ ----- Training and Validation Finished! ----- ✅\n\n\n")
        training_end_time = time.time()
        total_time = training_end_time - training_start_time
        print(f"===== Total time taken in training and validation:{total_time} Sec =====\n")
        
        # Plot losses and save PNG
        plot_losses_path =  os.path.join(project_root, f"Outputs/Visual")
        Evaluation.plot_losses(train_losses, val_losses, save_dir = plot_losses_path, filename = f"{path}.PNG", show = False)
        
        # Inference
        model = Utils.load_model(model, device, save_best_model_path)
        forecast = trainer.predict(test_loader, model, device)
        inversed_forecast = data_prep.inverse_target(forecast.cpu().numpy())
        inverse_actual = data_prep.inverse_target(y_test.cpu().numpy())
        inverse_input =  data_prep.inverse_target(X_test[:, :, 0].cpu().numpy()) 

        # Save Inference
        inference_path = os.path.join(project_root, f"Outputs/Inference")
        Utils.save_inference(inference_path, inverse_input, inverse_actual, inversed_forecast, X_test_time, y_test_time, name = path)

        # Evaluation results
        performance = Evaluation.calculatePerformance(inverse_actual, inversed_forecast, train_data['RRP'].values, test_data['RRP'].values, y_test_time)

        # Save Results
        performance_log_path = os.path.join(project_root, f"Outputs/Performance/{dataset_id}/Results.csv")
        performance_log = Evaluation.save_results_log(performance_log_path, performance, model_id, dataset_id, features_dim, granularity, 
                                                    scale_method, str(loss_function)[:-2], input_seq_len, output_seq_len, total_time, seed,
                                                    trainer.best_epoch, trainer.total_epoch, trainer.best_val_loss_overall)
    
        # Print Results
        print("************************  Results  ************************\n")
        for key, value in performance_log.items():
            print(f"'{key}': {value}")
        print("\n***********************************************************")