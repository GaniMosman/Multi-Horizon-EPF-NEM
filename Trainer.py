import copy
import torch
import pandas as pd
from tqdm import tqdm
from Modules import Utils


class Trainer:
    
    def __init__(self):
        self.best_val_loss_for_stop = float('inf')  
        self.best_val_loss_overall = float('inf') 
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = None
        self.total_epoch = None
        
    # -------------------------
    # Early Stopping Check
    # -------------------------
    def early_stopping(self, val_loss, patience=5, min_delta=0.0, verbose=True):
        stop = False
        if val_loss < self.best_val_loss_for_stop - min_delta:
            self.best_val_loss_for_stop  = val_loss
            self.patience_counter = 0
            if verbose:
                print(f"🟢 Validation loss improved to {val_loss:.6f}")
            stop = False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                if verbose:
                    print(f"++ Early stopping triggered after {self.patience_counter} epochs without improvement ++")
                stop = True
        return stop
    
    # -------------------------
    # Training Step Function
    # -------------------------  
    def train_step(self, data, model, loss_function, optimizer, device): 
        total_samples = 0
        total_loss = 0
        batch_losses = []  
        model.train()
        
        for batch in data:
            batch = [tensor.to(device) for tensor in batch]
            y_true = batch.pop(-1)
            inputs = batch
            output = model(*inputs)

            loss = loss_function(output, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            
            batch_size = y_true.size(0)
            total_loss += batch_loss * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        return avg_loss, batch_losses

     
    # -------------------------
    # Validation Function
    # -------------------------
    def validate(self, data, model, loss_function, device):
        total_loss = 0
        total_samples = 0
        batch_losses = []
        model.eval()
        
        with torch.no_grad():
            for batch in data:
                batch = [tensor.to(device) for tensor in batch]
                y_true = batch.pop(-1)
                inputs = batch
                output = model(*inputs)
                loss = loss_function(output, y_true)
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                batch_size = y_true.size(0)
                total_loss += batch_loss * batch_size
                total_samples += batch_size
            
        avg_loss = total_loss / total_samples
        return avg_loss, batch_losses
    

    # -------------------------
    # Prediction Function
    # -------------------------   
    def predict(self, data, model, device):
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in data:
                batch = [tensor.to(device) for tensor in batch]
                inputs = batch[:-1] 
                y_hat = model(*inputs)
                preds.append(y_hat)
        return torch.cat(preds, dim=0)

    # -------------------------
    # Training Function
    # -------------------------
    def train(self, model, train_data, val_data, epochs, loss_function, optimizer,
              device, early_stopping_patience=None, min_delta=0.0, lr_scheduler = None,  
              save_best_model_path=None, log_csv_path=None, verbose=True):

        train_losses = []
        val_losses = []
        
        if verbose:
            iterator = range(1, epochs + 1)
        else:
            iterator = tqdm(range(1, epochs + 1), desc="Train ~ Validation")
        
        for epoch in iterator:
            self.total_epoch = epoch

            train_loss, _ = self.train_step(train_data, model, loss_function, optimizer, device)
            val_loss, _ = self.validate(val_data, model, loss_function, device)

            if verbose:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Track overall best model (for saving)
            if val_loss < self.best_val_loss_overall:
                self.best_val_loss_overall = val_loss
                self.best_epoch = epoch
                if save_best_model_path is not None:
                    self.best_model_state = copy.deepcopy(model.state_dict())
            
            # Early stopping check
            if early_stopping_patience is not None:
                stop = self.early_stopping(
                    val_loss, patience=early_stopping_patience, 
                    min_delta=min_delta, verbose=verbose
                )
                if stop:
                    break  
            
            # Adaptive learning rate
            if lr_scheduler is not None and epoch < epochs:
                old_lr = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr and verbose:
                    print(f"🔽 LR updated: {old_lr:.5f} -> {new_lr:.5f}")

        # Save best (lowest validation loss) model
        if save_best_model_path is not None and self.best_model_state is not None:
            print(f"\n****** Best Validation loss: {self.best_val_loss_overall:.6f} achieved at epoch {self.best_epoch} ******\n")
            Utils.save_model(self.best_model_state, save_best_model_path)

        # Save training log to a csv file
        if log_csv_path is not None:
            Utils.save_training_log(log_csv_path, train_losses, val_losses)
           
        return train_losses, val_losses

   