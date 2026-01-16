import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error


class Evaluation:

    @staticmethod
    def plot_losses(train_losses, val_losses, save_dir, filename, show=False):
       
        df = pd.DataFrame({
            "Epoch": list(range(1, len(train_losses)+1)) * 2,
            "Loss": train_losses + val_losses,
            "Type": ["Training"] * len(train_losses) + ["Validation"] * len(val_losses)
        })

        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("Set1", 2)[::-1]

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x="Epoch", y="Loss", hue="Type", 
            palette=palette, linewidth=2.5, marker="o", markersize=7
        )

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(1, len(train_losses)) 

        sns.despine()
        
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(title="", fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=600, bbox_inches="tight")

        if show:
            plt.title("Training vs Validation Loss", fontsize=16, pad = 25, weight="bold")
            plt.show()
        else:
            plt.close()
        
        print(f"✅ Loss Plot saved to {save_path}!\n")


    @staticmethod
    def plot_forecast(timestamps, actual, predictions, num_plots=3, start_index=0):
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("Set2", 2)

        n_samples, horizon = actual.shape

        start_index = max(0, start_index)
        end_index = min(start_index + num_plots, n_samples)
        indices = list(range(start_index, end_index))
        num_plots = len(indices)

        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=False)
        if num_plots == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.plot(timestamps[idx], actual[idx], label="Actual", color=palette[0], linewidth=2)
            ax.plot(timestamps[idx], predictions[idx], label="Forecast", color=palette[1], linewidth=2, alpha=0.8, linestyle="--")

            start_time = timestamps[idx][0]
            end_time = timestamps[idx][-1]
            ax.set_title(
                f"{start_time} → {end_time}, Forecast Sequence Length: {horizon}, Horizon: {int(horizon/2)}H",
                fontsize=10, pad=20, fontweight="bold"
            )

            ax.set_xlabel("Timestamp", fontsize=12)
            ax.set_ylabel("Price (AU$)", fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True)

            ax.tick_params(axis="x", rotation=30)
            sns.despine(ax=ax)

        plt.subplots_adjust(hspace=1)
        plt.show()

    @staticmethod
    def find_negative_prices(data):
        negative_mask = np.any(data < 0, axis=1)
        negative_indices = np.where(negative_mask)[0]
        
        return negative_indices.tolist()
    
    @staticmethod
    def find_extreme_high_prices(data, factor=3):
        deseq_data = np.concatenate([data[0], data[1:, -1]])
        mean = np.mean(deseq_data)
        std = np.std(deseq_data)
        threshold = mean + factor * std

        high_mask = np.any(data > threshold, axis=1)
        high_indices = np.where(high_mask)[0]

        return high_indices.tolist(), threshold
    
    @staticmethod
    def smape(y_true, y_pred):
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        smape_values = np.where(denominator == 0, 0, numerator / denominator)
        return np.mean(smape_values) * 100

    @staticmethod
    def smape_by_time_of_day(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        y_true_flat = np.asarray(y_true).flatten()
        y_pred_flat = np.asarray(y_pred).flatten()
        timestamps_flat = pd.to_datetime(np.asarray(timestamps).flatten())
    
        if len(timestamps_flat) < 2:
            return np.array([])
            
        delta_min = (timestamps_flat[1] - timestamps_flat[0]).total_seconds() / 60
        slots_per_day = int(np.round(24 * 60 / delta_min))
        time_of_day_idx = ((timestamps_flat.hour * 60 + timestamps_flat.minute) // delta_min).astype(int)
    
        numerator = np.abs(y_pred_flat - y_true_flat)
        denominator = (np.abs(y_true_flat) + np.abs(y_pred_flat)) / 2
        
        smape_values = np.where(denominator == 0, 0, numerator / denominator)
        
        smape_by_slot_raw: List[list] = [[] for _ in range(slots_per_day)]
        for i, slot in enumerate(time_of_day_idx):
            smape_by_slot_raw[slot].append(smape_values[i])
    
        smape_by_slot = np.array([np.mean(err) * 100 if len(err) > 0 else np.nan for err in smape_by_slot_raw])
        
        return smape_by_slot

    
    @staticmethod
    def mase(actuals, forecasts, insample, frequency=336):
        forecasts = np.asarray(forecasts)
        actuals   = np.asarray(actuals)
        insample  = np.asarray(insample)

        if forecasts.shape != actuals.shape:
            raise ValueError("Forecasts and actuals must have the same shape")
        
        if len(insample) <= frequency:
            raise ValueError("Insample data must be longer than the seasonal frequency")

        q = np.abs(insample[frequency:] - insample[:-frequency])
        scale = np.mean(q)
        if scale == 0:
            raise ValueError("Cannot compute MASE because the in-sample series is constant")

        abs_errors = np.abs(forecasts - actuals)

        mase_value = np.mean(abs_errors) / scale
        return mase_value
    
    @staticmethod
    def mase_by_time_of_day(actuals, forecasts, insample, timestamps, frequency=336):
        forecasts = np.asarray(forecasts)
        actuals = np.asarray(actuals)
        insample = np.asarray(insample)
        timestamps_flat = pd.to_datetime(timestamps.flatten())

        if forecasts.shape != actuals.shape:
            raise ValueError("Forecasts and actuals must have the same shape")

        delta_min = (timestamps_flat[1] - timestamps_flat[0]).total_seconds() / 60
        slots_per_day = int(24 * 60 / delta_min)
        
        if len(insample) <= frequency:
            raise ValueError("Insample series must be longer than the seasonal frequency")
        q = np.abs(insample[frequency:] - insample[:-frequency])
        scale = np.mean(q)
        if scale == 0:
            raise ValueError("Cannot compute MASE because the in-sample series is constant")
            
        actuals_flat = actuals.flatten()
        forecasts_flat = forecasts.flatten()
        abs_scaled_errors = np.abs(forecasts_flat - actuals_flat) / scale
        
        time_of_day_idx = ((timestamps_flat.hour * 60 + timestamps_flat.minute) // delta_min).astype(int)
        errors_by_slot = [[] for _ in range(slots_per_day)]
        for i, slot in enumerate(time_of_day_idx):
            errors_by_slot[slot].append(abs_scaled_errors[i])
        mase_by_slot = np.array([np.mean(err) if len(err) > 0 else np.nan for err in errors_by_slot])
        return mase_by_slot


    @staticmethod
    def rmae(actuals, forecasts, outsample, frequency=336):
        forecasts = np.asarray(forecasts)
        actuals   = np.asarray(actuals)
        outsample  = np.asarray(outsample)

        if forecasts.shape != actuals.shape:
            raise ValueError("Forecasts and actuals must have the same shape")
        
        if len(outsample) <= frequency:
            raise ValueError("Outsample data must be longer than the seasonal frequency")

        q = np.abs(outsample[frequency:] - outsample[:-frequency])
        benchmark = np.mean(q)
        if benchmark == 0:
            raise ValueError("Cannot compute rMAE because the outsample series is constant")

        abs_errors = np.abs(forecasts - actuals)

        rmae_value = np.mean(abs_errors) / benchmark
        return rmae_value

    @staticmethod
    def rmae_by_time_of_day(actuals, forecasts, outsample, timestamps, frequency=336):
        forecasts = np.asarray(forecasts)
        actuals = np.asarray(actuals)
        outsample = np.asarray(outsample)
        timestamps_flat = pd.to_datetime(timestamps.flatten())

        if forecasts.shape != actuals.shape:
            raise ValueError("Forecasts and actuals must have the same shape")

        delta_min = (timestamps_flat[1] - timestamps_flat[0]).total_seconds() / 60
        slots_per_day = int(24 * 60 / delta_min)
        
        if len(outsample) <= frequency:
            raise ValueError("Outsample series must be longer than the seasonal frequency")
        q = np.abs(outsample[frequency:] - outsample[:-frequency])
        benchmark = np.mean(q)
        if benchmark == 0:
            raise ValueError("Cannot compute rMAE because the outsample series is constant")
            
        actuals_flat = actuals.flatten()
        forecasts_flat = forecasts.flatten()
        abs_scaled_errors = np.abs(forecasts_flat - actuals_flat) / benchmark
        
        time_of_day_idx = ((timestamps_flat.hour * 60 + timestamps_flat.minute) // delta_min).astype(int)
        errors_by_slot = [[] for _ in range(slots_per_day)]
        for i, slot in enumerate(time_of_day_idx):
            errors_by_slot[slot].append(abs_scaled_errors[i])
        rmae_by_slot = np.array([np.mean(err) if len(err) > 0 else np.nan for err in errors_by_slot])
        return rmae_by_slot

    @staticmethod
    def mae_by_time_of_day(y_true, y_pred, timestamps):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        timestamps_flat = pd.to_datetime(timestamps.flatten())
        
        delta_min = (timestamps_flat[1] - timestamps_flat[0]).total_seconds() / 60
        slots_per_day = int(24 * 60 / delta_min)
        time_of_day_idx = ((timestamps_flat.hour * 60 + timestamps_flat.minute) // delta_min).astype(int)
        
        errors_by_slot = [[] for _ in range(slots_per_day)]
        for i, slot in enumerate(time_of_day_idx):
            errors_by_slot[slot].append(abs(y_true_flat[i] - y_pred_flat[i]))

        mae_by_slot = np.array([np.mean(err) if len(err) > 0 else np.nan for err in errors_by_slot])
        
        return mae_by_slot

    @staticmethod
    def rmse_by_time_of_day(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        y_true_flat = np.asarray(y_true).flatten()
        y_pred_flat = np.asarray(y_pred).flatten()
        timestamps_flat = pd.to_datetime(np.asarray(timestamps).flatten())
        
        if len(timestamps_flat) < 2:
            return np.array([])
            
        delta_min = (timestamps_flat[1] - timestamps_flat[0]).total_seconds() / 60
        slots_per_day = int(np.round(24 * 60 / delta_min))
    
        time_of_day_idx = ((timestamps_flat.hour * 60 + timestamps_flat.minute) // delta_min).astype(int)
        
        sq_errors = (y_true_flat - y_pred_flat) ** 2
        
        sq_errors_by_slot: List[list] = [[] for _ in range(slots_per_day)]
        for i, slot in enumerate(time_of_day_idx):
            sq_errors_by_slot[slot].append(sq_errors[i])
    
        rmse_by_slot = np.array([
            np.sqrt(np.mean(err)) if len(err) > 0 else np.nan 
            for err in sq_errors_by_slot
        ])
        
        return rmse_by_slot

    @staticmethod
    def mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
    
        actual_change = np.diff(y_true, axis=-1)
        predicted_change = np.diff(y_pred, axis=-1)
        
        actual_direction = np.sign(actual_change)
        predicted_direction = np.sign(predicted_change)
        
        non_zero_actual_changes = actual_direction != 0
        
        correct_predictions = (actual_direction == predicted_direction) & non_zero_actual_changes
    
        num_correct = np.sum(correct_predictions)
        num_total_evaluations = np.sum(non_zero_actual_changes)
    
        if num_total_evaluations == 0:
            return 0.0 
        mda = (num_correct / num_total_evaluations) * 100
        return mda
        
    @staticmethod
    def mda_by_time_of_day(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        timestamps = np.asarray(timestamps)
        
        if y_true.shape != y_pred.shape or y_true.shape != timestamps.shape:
            raise ValueError("y_true, y_pred, and timestamps must have the same shape.")
    
        true_diff = np.diff(y_true, axis=1).flatten()
        pred_diff = np.diff(y_pred, axis=1).flatten()
        try:
            timestamps_diff_flat = pd.to_datetime(timestamps[:, 1:].flatten())
        except IndexError:
            print("Warning: Input data shape is too small to calculate differences (n_timesteps < 2).")
            return np.array([])
    
        if len(timestamps_diff_flat) < 1:
            return np.array([])
            
        try:
            delta_min = (timestamps_diff_flat[1] - timestamps_diff_flat[0]).total_seconds() / 60
        except IndexError:
            print("Warning: Only one data point remains after differencing. Cannot infer delta_min automatically.")
            return np.array([])
    
        if delta_min <= 0:
            raise ValueError(f"Calculated time step duration is non-positive ({delta_min} min). Check timestamp data integrity.")
    
        slots_per_day = int(np.round(24 * 60 / delta_min))
        if slots_per_day == 0:
             return np.array([])
    
        time_of_day_idx = ((timestamps_diff_flat.hour * 60 + timestamps_diff_flat.minute) // delta_min).astype(int)
        
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        
        actual_change_occurred = (true_direction != 0)
        
        correct_predictions = (true_direction == pred_direction) & actual_change_occurred
        
        correct_counts = np.zeros(slots_per_day, dtype=int)
        total_counts = np.zeros(slots_per_day, dtype=int)
    
        for i, slot in enumerate(time_of_day_idx):
            if actual_change_occurred[i]:
                total_counts[slot] += 1
                if correct_predictions[i]:
                    correct_counts[slot] += 1
    
        mda_by_slot = np.full(slots_per_day, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            mda_values = (correct_counts / total_counts) * 100
            
            valid_indices = total_counts > 0
            mda_by_slot[valid_indices] = mda_values[valid_indices]
    
        return mda_by_slot

    @staticmethod
    def calculatePerformance(actual, prediction, insample, outsample, y_time):
        return {
            'MAE': mean_absolute_error(actual, prediction),
            'MAE_per_step': Evaluation.mae_by_time_of_day(actual, prediction, np.array(y_time)),
            'RMSE' : root_mean_squared_error(actual, prediction),
            'RMSE_per_step' : Evaluation.rmse_by_time_of_day(actual, prediction, np.array(y_time)),
            'sMAPE': Evaluation.smape(actual, prediction),
            'sMAPE_per_step': Evaluation.smape_by_time_of_day(actual, prediction, np.array(y_time)),
            'MASE' : Evaluation.mase(actual, prediction, insample),
            'MASE_per_step': Evaluation.mase_by_time_of_day(actual, prediction, insample, np.array(y_time)),
            'rMAE' : Evaluation.rmae(actual, prediction, outsample),
            'rMAE_per_step': Evaluation.rmae_by_time_of_day(actual, prediction, outsample, np.array(y_time)),
            'MDA': Evaluation.mda(actual, prediction),
            'MDA_per_step': Evaluation.mda_by_time_of_day(actual, prediction, np.array(y_time))
        }
    
    @staticmethod
    def save_results_log(performance_log_path, metrics, model_id, dataset_id, features_dim,
                        granularity, scale_method, loss_function, input_seq_len, output_seq_len,
                        total_time, seed, best_epoch, total_epoch, best_val_loss_overall):
        meta = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'RunID': int(datetime.now().strftime("%Y%m%d%H%M%S")),
        'Model': model_id,
        'Dataset': dataset_id,
        'Features_dim': features_dim,
        'Granularity': granularity,
        'Scale_method': scale_method,
        'Loss_function': loss_function,
        'Input_len': input_seq_len,
        'Output_len': output_seq_len,
        'Horizon': str(int(output_seq_len/2)) + 'H',
        'Training_time(s)': total_time,
        'Seed': seed,
        'Best_epoch': best_epoch,
        'Total_epoch': total_epoch,
        'Best_loss': best_val_loss_overall
        }

        metrics_with_meta = {**meta, **metrics}
        
        df_new = pd.DataFrame([metrics_with_meta])

        if os.path.exists(performance_log_path):
            df_new.to_csv(performance_log_path, mode='a', index=False, header=False)
        else:
            df_new.to_csv(performance_log_path, mode='w', index=False, header=True)
        
        print(f"✅ Results saved to {performance_log_path}\n")
        return metrics_with_meta

    @staticmethod
    def load_results(performance_log_path):
        if os.path.exists(performance_log_path):
            return pd.read_csv(performance_log_path)
        else:
            print("⚠️ No results file found yet.")
            return pd.DataFrame()