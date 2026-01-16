import torch
import holidays
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class DataPreparation:
    def __init__(self):
        self.scaler = None
        self.target_col = None

    def downsample_data(self, data, freq='1h', agg_by='mean'):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Resample Error: DataFrame index must be a DateTimeIndex")

        data = data.fillna(0)
        data = data.sort_index()
        data = data.resample(freq, closed = 'left', label = 'left').agg(agg_by)
    
        return data

    def add_holiday_column(self, data, state_code):
        au_holidays = holidays.AU(years=range(2022, 2026), state=state_code)
        data['is_holiday'] = data.index.to_series().apply(
            lambda date: 1 if date.date() in au_holidays else 0
        )
        return data

    def split_time_series(self, data, train_ratio=0.8, cutoff_date=None, val_ratio=0.3):

        data = data.fillna(0)
        data = data.sort_index()
        
        if cutoff_date is not None:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex for date-based splitting.")
            cutoff_date = pd.to_datetime(cutoff_date)
            train = data[data.index <= cutoff_date]
            test = data[data.index > cutoff_date]
        else:
            n = len(data)
            train_size = int(n * train_ratio)
            train, test = data.iloc[:train_size], data.iloc[train_size:]

        if val_ratio > 0:
            val_size = int(len(train) * val_ratio)
            if val_size > 0:
                train, val = train.iloc[:-val_size], train.iloc[-val_size:]
            else:
                val = None
        else:
            val = None

        return train, val, test

    def extract_temporal_features(self, df_index: pd.DatetimeIndex) -> pd.DataFrame:
        if not isinstance(df_index, pd.DatetimeIndex):
            raise TypeError("Input must be a pandas DatetimeIndex.")
    
        features_df = pd.DataFrame({
            'month': df_index.month,
            'day': df_index.day,
            'weekday': df_index.dayofweek,
            'hour': df_index.hour
        })
        
        features_df.index = df_index
        
        return features_df

    def scale(self, train, val=None, test=None, method="MinMax", target_col=None):
        
        if method == "MinMax":
            scaler = MinMaxScaler()
        elif method == "Zscore":
            scaler = StandardScaler()
        elif method == "Robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'MinMax', 'Zscore' or 'Robust'")
        
        if target_col is not None:
            self.scaler = scaler
            self.target_col = target_col
        
        if isinstance(train, pd.DataFrame):
            train_scaled = pd.DataFrame(scaler.fit_transform(train),
                                        index=train.index, columns=train.columns)
            val_scaled   = (pd.DataFrame(scaler.transform(val),
                                         index=val.index, columns=val.columns)
                            if val is not None else None)
            test_scaled  = (pd.DataFrame(scaler.transform(test),
                                         index=test.index, columns=test.columns)
                            if test is not None else None)
        else:
            train_scaled = scaler.fit_transform(train)
            val_scaled   = scaler.transform(val) if val is not None else None
            test_scaled  = scaler.transform(test) if test is not None else None
        
        return train_scaled, val_scaled, test_scaled

    def inverse(self, data):

        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call scale() first.")

        if isinstance(data, pd.DataFrame):
            inv = self.scaler.inverse_transform(data)
            return pd.DataFrame(inv, index=data.index, columns=data.columns)

        else:
            return self.scaler.inverse_transform(data)
        
        
    def inverse_target(self, y_scaled):
        
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call scale() first.")

        n_features = self.scaler.n_features_in_
        target_idx = self.target_col
    
        if y_scaled.ndim == 1:
            dummy = np.zeros((len(y_scaled), n_features))
            dummy[:, target_idx] = y_scaled.reshape(-1)
            inv = self.scaler.inverse_transform(dummy)
            return inv[:, target_idx]
    
        elif y_scaled.ndim == 2:
            n_samples, n_steps = y_scaled.shape
            inv_steps = np.zeros_like(y_scaled)
            for step in range(n_steps):
                dummy = np.zeros((n_samples, n_features))
                dummy[:, target_idx] = y_scaled[:, step]
                inv = self.scaler.inverse_transform(dummy)
                inv_steps[:, step] = inv[:, target_idx]
            return inv_steps
    
        else:
            raise ValueError("y_scaled must be 1D or 2D array")


    def inverse_input(self, X_scaled):

        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call scale() first.")
        if self.target_col is None:
            raise ValueError("target_col must be set before calling inverse_input().")

        n_features = self.scaler.n_features_in_
        non_target_idx = [i for i in range(n_features) if i != self.target_col]

        if isinstance(X_scaled, pd.DataFrame):
            X_full = X_scaled.copy()
           
            X_full.insert(self.target_col, "target_dummy", 0.0)
            inv = pd.DataFrame(self.scaler.inverse_transform(X_full),
                            index=X_full.index,
                            columns=X_full.columns)
          
            inv = inv.drop(inv.columns[self.target_col], axis=1)
            return inv

        else:
            
            if X_scaled.ndim == 1:
                X_scaled = X_scaled.reshape(-1, 1)
            
            if X_scaled.ndim == 2:
                n_samples = X_scaled.shape[0]
                dummy = np.zeros((n_samples, n_features))
                dummy[:, non_target_idx] = X_scaled
                inv = self.scaler.inverse_transform(dummy)
                return inv[:, non_target_idx]

            elif X_scaled.ndim == 3:
                n_samples, n_steps, n_in_features = X_scaled.shape
                inv_out = np.zeros_like(X_scaled, dtype=float)

                for t in range(n_steps):
                    dummy = np.zeros((n_samples, n_features))
                    dummy[:, non_target_idx] = X_scaled[:, t, :]
                    inv = self.scaler.inverse_transform(dummy)
                    inv_out[:, t, :] = inv[:, non_target_idx]
                return inv_out

            else:
                raise ValueError("X_scaled must be 1D, 2D, or 3D array")


    def create_sequences(self, data, input_length, output_length, stride=1, target_col=None, index=None):
        
        if not torch.is_tensor(data):
            data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data, dtype=torch.float32)

        if data.ndim == 1:
            data = data.unsqueeze(-1)
        
        X, y, X_idx, y_idx = [], [], [], []
        total_length = input_length + output_length

        if total_length > len(data):
            raise ValueError("Combined input and output length exceeds data size.")
        
        for i in range(0, len(data) - total_length + 1, stride):
            seq_x = data[i:i + input_length]
            X.append(seq_x)

            if target_col is not None:
                seq_y = data[i + input_length:i + total_length, target_col]
                y.append(seq_y)

            if index is not None:
                X_idx.append(index[i:i + input_length])
                if target_col is not None:
                    y_idx.append(index[i + input_length:i + total_length])

        X = torch.stack(X)
        y = torch.stack(y) if y else None

        if target_col is not None:
            return X, y, X_idx, y_idx
        else:
            return X, X_idx

    def create_dataloader(self, *args, batch_size=32, shuffle=False):
        dataset = TensorDataset(*args)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)