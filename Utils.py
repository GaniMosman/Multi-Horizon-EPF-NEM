import os
import sys
import json
import h5py
import torch
import random
import joblib
import importlib
import numpy as np
import pandas as pd
from datetime import datetime

def set_seed(seed: int = 42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_splits(path, train, val, test, name):

    os.makedirs(path, exist_ok=True)

    if train is not None:
        train.to_parquet(os.path.join(path, f"train_{name}.parquet"))
    if val is not None:
        val.to_parquet(os.path.join(path, f"val_{name}.parquet"))
    if test is not None:
        test.to_parquet(os.path.join(path, f"test_{name}.parquet"))

    print(f"✅ Train, Test, Validation sets saved to {path}!\n")

def load_splits(path):
    data = pd.read_parquet(path)
    return data

def save_sequences(path, X, y, X_idx, y_idx, name):

    os.makedirs(path, exist_ok=True)

    file_name = f"{name}.h5"
    file_path = os.path.join(path, file_name)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("X", data=X.numpy())
        f.create_dataset("y", data=y.numpy())

        X_str = np.array([[str(ts) for ts in idx] for idx in X_idx], dtype="S")
        y_str = np.array([[str(ts) for ts in idx] for idx in y_idx], dtype="S")

        f.create_dataset("X_idx", data=X_str)
        f.create_dataset("y_idx", data=y_str)

    print(f"✅ Sequences saved to {file_path}!")


def load_sequences(file_path):

    with h5py.File(file_path, "r") as f:
        X = torch.from_numpy(f["X"][:])
        y = torch.from_numpy(f["y"][:])

        X_idx = [[pd.to_datetime(ts.decode()) for ts in seq] for seq in f["X_idx"][:]]
        y_idx = [[pd.to_datetime(ts.decode()) for ts in seq] for seq in f["y_idx"][:]]
    
    print(f"✅ Sequences loaded from {file_path}!")
    return X, y, X_idx, y_idx

def save_temp_sequences(path, seq_tensor, name):

    os.makedirs(path, exist_ok=True)

    file_name = f"{name}_temp.h5"
    file_path = os.path.join(path, file_name)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("temp_seq", data=seq_tensor.numpy())

    print(f"✅ Temporal sequences saved to {file_path}!")

def load_temp_sequences(file_path):
    with h5py.File(file_path, "r") as f:
        temp_seq = torch.from_numpy(f["temp_seq"][:])

    print(f"✅ Temporal sequences loaded from {file_path}!")
    return temp_seq


def save_scaler(scaler, save_dir, filename):

    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    joblib.dump(scaler, filepath)
    print(f"✅ Scaler saved to {filepath}!\n")

def load_scaler(save_dir, filename):
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No scaler file found at {filepath}!\n")
    return joblib.load(filepath)


def return_best_params(results_df):

    results_df_sorted = results_df.sort_values(
        by=["best_val_loss", "avg_val_loss"], 
        ascending=[True, True]
    )
    
    best_row = results_df_sorted.iloc[0]
    best_params = best_row.to_dict()
    best_loss = best_params.pop("best_val_loss")
    
    return best_params, best_loss


def save_best_params(output_dir, model_key, best_params):

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{model_key}.json")

    history = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": best_params
    }
    history.append(entry)

    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\n✅ Saved best hyperparameters for {model_key}!")
    

def load_best_params(output_dir, model_key):
 
    json_path = os.path.join(output_dir, f"{model_key}.json")

    if not os.path.exists(json_path):
        msg = f"{model_key}.json not found at {output_dir}, ** using default parameters! **\n"
        print(f"⚠️ {msg}")
        return {}

    try:
        with open(json_path, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        msg = f"Corrupted JSON at {json_path}!"
        print(f"❌ {msg}")
        return {}

    if not history:
        msg = f"Empty history for {model_key}, ** using default parameters! **\n"
        print(f"⚠️ {msg}")
        return {}

    latest_entry = history[-1]
    print(f"✅ Loaded args from {model_key}!\n")
    print(f" Parameters: {latest_entry["params"]}\n")

    return latest_entry["params"]


def save_training_log(log_csv_path, train_losses, val_losses):
    log_df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses)+1)),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
    log_df.to_csv(log_csv_path, index=False)
    print(f"✅ Training log saved to {log_csv_path}\n")


def save_inference(path, X, y_true, y_pred, X_idx, y_idx, name):

    os.makedirs(path, exist_ok=True)

    file_name = f"{name}.h5"
    file_path = os.path.join(path, file_name)

    X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.array(X)
    y_true_np = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else np.array(y_true)
    y_pred_np = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else np.array(y_pred)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("Lookback", data=X_np)
        f.create_dataset("Actual", data=y_true_np)
        f.create_dataset("Forecast", data=y_pred_np)

        X_str = np.array([[str(ts) for ts in idx] for idx in X_idx], dtype="S")
        y_str = np.array([[str(ts) for ts in idx] for idx in y_idx], dtype="S")

        f.create_dataset("Lookback_Timestamp", data=X_str)
        f.create_dataset("Forecast_Timestamp", data=y_str)

    print(f"✅ Inference saved to {file_path}!\n")


def load_inference(file_path):
    with h5py.File(file_path, "r") as f:

        X = torch.from_numpy(f["Lookback"][:])
        y_true = torch.from_numpy(f["Actual"][:])
        y_pred = torch.from_numpy(f["Forecast"][:])

        X_windows, X_seq_len = X.shape 
        y_windows, y_seq_len = y_true.shape 

        X_ts_bytes = f["Lookback_Timestamp"][:].flatten()
        y_ts_bytes = f["Forecast_Timestamp"][:].flatten()

        X_ts_str = X_ts_bytes.astype(str)
        y_ts_str = y_ts_bytes.astype(str)

        X_idx_flat = pd.to_datetime(X_ts_str)
        y_idx_flat = pd.to_datetime(y_ts_str)

        X_idx = X_idx_flat.values.reshape(X_windows, X_seq_len)
        y_idx = y_idx_flat.values.reshape(y_windows, y_seq_len)

    return X, y_true, y_pred, X_idx, y_idx

 # -------------------------
# Save Model Function
# -------------------------
def save_model(best_model_state, filepath):

    if best_model_state is None:
        print("❌ No model state to save!\n")
        return
    try:
        torch.save(best_model_state, filepath)
        print("┌---------------------------------------------┐")
        print("|          Best Model successfully saved!     |")
        print("└---------------------------------------------┘\n\n")
    except Exception as e:
        print(f"❌ Error saving model: {e}\n")

        
# -------------------------
# Load Model Function
# -------------------------
def load_model(model, device, filepath):
    try:
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Model successfully loaded from {filepath}!\n")
        return model
    except FileNotFoundError:
        print(f"❌ Error: The file {filepath} was not found!\n")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}!\n")
        return None


def get_model_class_and_args(model_id, X_train, y_train, Models):
  
    # --- Try to load directly from Modules.Models ---
    try:
        model_class = getattr(Models, model_id)
        min_args = {
            "num_features": X_train.shape[2],
            "output_seq_len": y_train.shape[1],
        }
        return model_class, min_args

    except AttributeError:
        # --- Load config for TSLib models ---
        base_dir = os.path.dirname(__file__)
        ts_dir = os.path.join(base_dir, "TSLib")
        config_path = os.path.join(ts_dir, "model_args.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ model_args.json not found at {config_path}")

        with open(config_path, "r") as f:
            TSLIB_MODEL_ARGS = json.load(f)

        model_class = None
        min_args = {}

        for file in os.listdir(ts_dir):
            if file.endswith(".py") and not file.startswith("__"):
                mod_name = file[:-3]
                mod = importlib.import_module(f"Modules.TSLib.{mod_name}")

                if hasattr(mod, model_id):
                    model_class = getattr(mod, model_id)
                    required_args = TSLIB_MODEL_ARGS.get(model_id, [])

                    if "num_features" in required_args:
                        min_args["num_features"] = X_train.shape[2]
                    if "seq_len" in required_args:
                        min_args["seq_len"] = X_train.shape[1]
                    if "pred_len" in required_args:
                        min_args["pred_len"] = y_train.shape[1]
                    break

        if model_class is None:
            print(f"❌ Model '{model_id}' not found in Modules.Models or Modules.TSLib!\n")
            sys.exit(1)

        return model_class, min_args