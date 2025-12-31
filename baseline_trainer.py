import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from tqdm import tqdm
import sys

# LibAUC imports for DAM (Deep AUC Maximization) with fallback
try:
    from libauc.losses import AUCMLoss
    from libauc.optimizers import PESG
    _LIBAUC_AVAILABLE = True
except Exception:
    _LIBAUC_AVAILABLE = False

# Import the correct modules from your project
from SupervisedBaselineDataLoader import SupervisedBaselineDataLoader
from baselinemodel import GCN_Classifier, BaselineOODModel

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Trains GNN backbone using SupervisedBaselineDataLoader for proper supervised learning.
    Uses real classification labels from the dataset.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else
                                 'cuda' if torch.cuda.is_available() else 'cpu')

        # Data loader will be initialized in train() to include encoding time in training time
        self.data_loader = None

        # Model will be created during training
        self.model = None

        # Initialize task type flags (will be set in train() after data loader is ready)
        self.is_regression = False
        self.is_multitask = False
        self.num_classes = None
        self.input_dim = None

    def _estimate_imratio_from_loader(self, loader):
        """Estimate positive class ratio from dataloader for DAM training."""
        pos, tot = 0, 0
        label_values = []
        for batch in loader:
            labels = batch['labels']
            if labels.ndim > 1:
                labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.argmax(dim=1)
            labels = labels.detach().cpu().numpy().reshape(-1)
            # Convert to {0,1} if using {-1,1} encoding
            if labels.min() < 0:
                labels = (labels > 0).astype('int')
            pos += (labels == 1).sum()
            tot += labels.shape[0]
            label_values.extend(labels.tolist())

        # Debug information
        unique_values = np.unique(label_values)
        logger.info(f"Label distribution - unique values: {unique_values}")
        logger.info(f"Label counts: {[(val, np.sum(np.array(label_values) == val)) for val in unique_values]}")

        imratio = (pos / max(tot, 1)) if tot > 0 else 0.5

        # Ensure minimum imratio for stable training
        if imratio < 0.01 or imratio > 0.99:
            logger.warning(f"Extreme imratio detected: {imratio:.4f}, adjusting to 0.1 for stable training")
            imratio = 0.1

        return float(imratio)

    def _determine_num_classes(self):
        """Determine the number of classes/tasks from the loaded labels."""
        try:
            # Check training labels to determine number of classes/tasks
            if hasattr(self.data_loader, 'final_labels') and 'train_id' in self.data_loader.final_labels:
                train_labels = self.data_loader.final_labels['train_id']
                if train_labels and len(train_labels) > 0:
                    sample_label = train_labels[0]

                    # Check if it's multi-task (array-like) or single task
                    if hasattr(sample_label, '__len__') and len(sample_label) > 1:
                        # Multi-task case - return number of tasks
                        num_tasks = len(sample_label)
                        logger.info(f"Multi-task learning detected: {num_tasks} tasks")
                        self.is_multitask = True
                        return num_tasks
                    else:
                        # Single task case - determine number of classes
                        self.is_multitask = False
                        if isinstance(sample_label, (list, np.ndarray)):
                            # Extract single values for analysis
                            single_labels = [l[0] if hasattr(l, '__len__') and len(l) > 0 else l for l in train_labels]
                        else:
                            single_labels = train_labels

                        unique_labels = set(single_labels)
                        num_classes = len(unique_labels)

                        # Check if this is a regression task
                        min_label, max_label = min(unique_labels), max(unique_labels)

                        # Detect regression task: continuous values or too many unique values
                        if (len(unique_labels) > 20 or
                            any(isinstance(label, float) and not label.is_integer() for label in unique_labels)):
                            logger.info(f"Detected regression task with {len(unique_labels)} unique values "
                                      f"in range [{min_label:.4f}, {max_label:.4f}]")
                            self.is_regression = True
                            return 1  # For regression, output dimension is 1

                        # Classification task
                        self.is_regression = False
                        if min_label < 0 or max_label >= num_classes:
                            logger.warning(f"Labels not in expected range [0, {num_classes-1}]: "
                                         f"found range [{min_label}, {max_label}]")
                            # Remap labels if necessary
                            num_classes = max_label + 1

                        return max(num_classes, 2)  # Ensure at least binary classification

            # Fallback to binary classification
            logger.warning("Could not determine number of classes from data, defaulting to binary")
            self.is_multitask = False
            return 2

        except Exception as e:
            logger.error(f"Error determining number of classes: {e}")
            self.is_multitask = False
            return 2

    def _calculate_pos_weights(self, train_loader):
        """Calculate positive weights for multi-task BCE loss balancing."""
        if not hasattr(self, 'is_multitask') or not self.is_multitask:
            return None

        try:
            import torch
            import numpy as np

            # Count positive/negative samples for each task
            task_counts = None
            total_samples = 0

            for batch in train_loader:
                labels = batch['labels']  # [batch_size, num_tasks]
                masks = batch.get('masks', torch.ones_like(labels, dtype=torch.bool))

                if task_counts is None:
                    num_tasks = labels.shape[1]
                    task_counts = {'pos': torch.zeros(num_tasks), 'neg': torch.zeros(num_tasks)}

                # Count valid positive/negative samples for each task
                for task_idx in range(labels.shape[1]):
                    task_labels = labels[:, task_idx]
                    task_mask = masks[:, task_idx]
                    valid_labels = task_labels[task_mask]

                    if len(valid_labels) > 0:
                        task_counts['pos'][task_idx] += (valid_labels > 0.5).sum()
                        task_counts['neg'][task_idx] += (valid_labels <= 0.5).sum()

                total_samples += labels.shape[0]

            # Calculate pos_weight = neg_count / pos_count for each task
            pos_weights = []
            for task_idx in range(len(task_counts['pos'])):
                pos_count = task_counts['pos'][task_idx].item()
                neg_count = task_counts['neg'][task_idx].item()

                if pos_count > 0:
                    pos_weight = neg_count / pos_count
                else:
                    pos_weight = 1.0  # Default if no positive samples

                pos_weights.append(pos_weight)

            pos_weights_tensor = torch.FloatTensor(pos_weights).to(self.device)
            logger.info(f"Pos weights range: {pos_weights_tensor.min():.3f} - {pos_weights_tensor.max():.3f}")

            return pos_weights_tensor

        except Exception as e:
            logger.warning(f"Failed to calculate pos_weights: {e}")
            return None

    def train(self):
        """Train the GNN backbone on ID data using supervised learning with real labels."""
        # Start timing (including data loading and feature encoding)
        train_start_time = time.time()
        epoch_times = []

        logger.info("Starting supervised baseline training with real labels...")

        # Initialize SUPERVISED data loader - this handles all data loading with real labels
        # Feature encoding time will be included in total training time (consistent with train.py)
        logger.info("Initializing SupervisedBaselineDataLoader...")
        self.data_loader = SupervisedBaselineDataLoader(self.args)

        # Determine dataset properties from loaded data
        self.num_classes = self._determine_num_classes()
        if hasattr(self, 'is_regression') and self.is_regression:
            logger.info(f"Detected regression task with output dimension {self.num_classes}")
        else:
            logger.info(f"Detected {self.num_classes} classes in the dataset")

        # Determine input dimension based on foundation model features
        if hasattr(self.data_loader, 'feature_cache') and self.data_loader.feature_cache:
            # Use foundation model features
            sample_features = next(iter(self.data_loader.feature_cache.values()))
            if isinstance(sample_features, torch.Tensor):
                self.input_dim = sample_features.shape[-1]
            else:
                self.input_dim = len(sample_features) if hasattr(sample_features, '__len__') else 512
            logger.info(f"Using foundation model features: dim={self.input_dim}")
        else:
            # This should not happen with SupervisedBaselineDataLoader, but fallback
            self.input_dim = 512
            logger.warning("Feature cache not available - this may indicate a problem")

        # Get encoding memory if available
        peak_encoding_memory_gb = getattr(self.data_loader, 'peak_encoding_memory_gb', 0.0)
        if peak_encoding_memory_gb > 0:
            logger.info(f"Foundation model encoding peak GPU memory: {peak_encoding_memory_gb:.2f} GB")

        # Reset GPU memory stats for training phase (after encoding)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Get DataLoaders from SupervisedBaselineDataLoader
        batch_size = getattr(self.args, 'batch_size', 32)
        num_workers = min(getattr(self.args, 'num_workers', 4), 2)

        train_loader = self.data_loader.get_training_loader(batch_size, num_workers)
        val_loader = self.data_loader.get_validation_loader(batch_size, num_workers)
        
        logger.info(f"Training loader: {len(train_loader.dataset)} samples")
        logger.info(f"Validation loader: {len(val_loader.dataset)} samples")
        
        # Initialize model
        hidden_dim = getattr(self.args, 'hidden_channels', 64)
        num_layers = getattr(self.args, 'num_layers', 3)
        dropout = getattr(self.args, 'dropout', 0.5)
        
        self.model = GCN_Classifier(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_classes,
            num_layers=num_layers,
            dropout=dropout,
            use_features=True  # Always true for SupervisedBaselineDataLoader
        ).to(self.device)
        
        logger.info(f"Model created: {sum(p.numel() for p in self.model.parameters())} parameters")

        # Check if DAM (Deep AUC Maximization) is enabled
        use_dam = str(getattr(self.args, 'method', '')).lower() in ('dam_msp', 'dam_energy', 'dam')

        if use_dam:
            if not _LIBAUC_AVAILABLE:
                raise RuntimeError("LibAUC not available. Install libauc>=1.3.0 or use the fallback surrogate.")
            if self.is_regression:
                raise ValueError("DAM expects classification; regression task not supported.")
            if self.is_multitask or self.num_classes > 2:
                logger.warning("DAM integration currently supports binary classification. "
                              "Multi-task/multi-class not supported in this minimal integration.")

            # Estimate imratio (based on training set)
            imratio = self._estimate_imratio_from_loader(train_loader)
            logger.info(f"DAM training enabled: imratio={imratio:.4f}, margin={getattr(self.args, 'dam_margin', 1.0)}")

            # LibAUC AUCMLoss + PESG
            aucm_loss = AUCMLoss(imratio=imratio, margin=getattr(self.args, 'dam_margin', 1.0))
            optimizer = PESG(self.model.parameters(),
                            loss_fn=aucm_loss,
                            lr=getattr(self.args, 'dam_lr', 0.1),
                            momentum=0.9,
                            weight_decay=getattr(self.args, 'weight_decay', 5e-4))
            criterion = aucm_loss
        else:
            # Setup optimizer and criterion for non-DAM methods
            lr = getattr(self.args, 'lr', 0.01)
            weight_decay = getattr(self.args, 'weight_decay', 5e-4)
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

            # Choose loss function based on task type (for non-DAM methods)
            if hasattr(self, 'is_multitask') and self.is_multitask:
                # Multi-task binary classification - use BCEWithLogitsLoss
                logger.info("Using BCEWithLogitsLoss for multi-task learning")

                # Calculate positive weights for class balancing
                pos_weights = self._calculate_pos_weights(train_loader)
                if pos_weights is not None:
                    logger.info(f"Calculated pos_weight for {len(pos_weights)} tasks")
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
                else:
                    criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                # Single task - either classification or regression
                if self.is_regression:
                    logger.info("Using MSELoss for regression")
                    criterion = nn.MSELoss()
                else:
                    logger.info("Using CrossEntropyLoss for single-task classification")
                    criterion = nn.CrossEntropyLoss()
        
        # Add _eval_val_auroc method for DAM validation
        def _eval_val_auroc(val_loader):
            self.model.eval()
            all_y, all_s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    logits = self.model(features)
                    if logits.size(1) == 1:
                        scores = logits.squeeze(-1)
                        y = labels
                        if y.ndim > 1: y = y.squeeze(-1)
                    else:
                        scores = logits[:, 1]
                        y = labels.argmax(dim=1) if labels.ndim > 1 else labels
                    all_s.append(scores.detach().cpu().numpy().reshape(-1))
                    all_y.append(y.detach().cpu().numpy().reshape(-1))
            all_s = np.concatenate(all_s)
            all_y = np.concatenate(all_y)
            # Convert to {0,1}
            if all_y.min() < 0: all_y = (all_y > 0).astype('int')

            # Debug information
            unique_y = np.unique(all_y)
            logger.debug(f"Validation labels - unique values: {unique_y}")
            logger.debug(f"Validation label counts: {[(val, np.sum(all_y == val)) for val in unique_y]}")
            logger.debug(f"Score range: [{all_s.min():.4f}, {all_s.max():.4f}]")

            try:
                # Check if we have both classes for valid AUROC calculation
                if len(unique_y) < 2:
                    logger.warning(f"Only {len(unique_y)} unique class(es) in validation set, cannot compute AUROC")
                    return 0.5
                auroc = float(roc_auc_score(all_y, all_s))
                return auroc if not np.isnan(auroc) else 0.5
            except Exception as e:
                logger.warning(f"AUROC calculation failed: {e}")
                return 0.5

        # Training loop
        epochs = getattr(self.args, 'epochs', 200)
        if use_dam:
            best_val_auc = 0.0
        else:
            best_val_acc = 0.0
        patience = getattr(self.args, 'patience', 20)
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
            for batch in train_pbar:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                masks = batch.get('masks', torch.ones_like(labels, dtype=torch.bool)).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)

                if use_dam:
                    # DAM loss calculation
                    if outputs.size(1) == 1:
                        y_pred = outputs.squeeze(-1)
                        y_true = labels
                        if y_true.ndim > 1:
                            y_true = y_true.squeeze(-1)
                    else:
                        # Binary classification take positive class (assuming label 1 is positive)
                        y_pred = outputs[:, 1]
                        y_true = labels
                        if y_true.ndim > 1:  # one-hot -> index
                            y_true = y_true.argmax(dim=1)
                    y_true = y_true.float()
                    loss = criterion(y_pred, y_true)
                else:
                    # Handle potential dimension mismatch
                    if outputs.size(1) != self.num_classes:
                        logger.error(f"Output dimension mismatch: model outputs {outputs.size(1)} classes, "
                                   f"but dataset has {self.num_classes} classes")
                        raise ValueError("Model-data dimension mismatch")

                    # Calculate loss based on task type
                    if hasattr(self, 'is_multitask') and self.is_multitask:
                        # Multi-task BCE loss with masking
                        raw_loss = criterion(outputs, labels)  # [batch_size, num_tasks]

                        # Apply mask to ignore missing labels
                        masked_loss = raw_loss * masks.float()

                        # Average over valid entries
                        valid_loss_sum = masked_loss.sum()
                        valid_count = masks.float().sum()

                        if valid_count > 0:
                            loss = valid_loss_sum / valid_count
                        else:
                            loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Single task - classification or regression
                        if self.is_regression:
                            # Regression task
                            if len(labels.shape) > 1:
                                labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.view(-1)
                            if labels.dtype != torch.float:
                                labels = labels.float()
                            outputs = outputs.squeeze(-1) if outputs.size(-1) == 1 else outputs
                            loss = criterion(outputs, labels)
                        else:
                            # Classification task
                            if len(labels.shape) > 1:
                                labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.argmax(dim=1)
                            if labels.dtype != torch.long:
                                labels = labels.long()
                            loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                # Calculate accuracy based on task type
                if hasattr(self, 'is_multitask') and self.is_multitask:
                    # Multi-task accuracy calculation
                    sigmoid_outputs = torch.sigmoid(outputs)
                    predicted = (sigmoid_outputs > 0.5).float()

                    # Only count accuracy for valid (non-masked) labels
                    correct = (predicted == labels).float() * masks.float()
                    train_correct += correct.sum().item()
                    train_total += masks.float().sum().item()
                else:
                    # Single task accuracy
                    if self.is_regression:
                        # For regression, calculate MAE
                        if len(labels.shape) > 1:
                            labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.view(-1)
                        outputs_squeezed = outputs.squeeze(-1) if outputs.size(-1) == 1 else outputs
                        mae = torch.abs(outputs_squeezed - labels).mean()
                        train_correct += mae.item() * labels.size(0)
                        train_total += labels.size(0)
                    else:
                        # Classification accuracy
                        pred = outputs.argmax(dim=1)
                        if len(labels.shape) > 1:
                            labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.argmax(dim=1)
                        train_correct += (pred == labels).sum().item()
                        train_total += labels.size(0)

                # Update progress bar
                current_acc = train_correct / train_total if train_total > 0 else 0
                metric_name = 'mae' if self.is_regression else 'acc'
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    metric_name: f'{current_acc:.4f}'
                })
            
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    masks = batch.get('masks', torch.ones_like(labels, dtype=torch.bool)).to(self.device)

                    outputs = self.model(features)

                    # Calculate validation loss based on task type
                    if hasattr(self, 'is_multitask') and self.is_multitask:
                        # Multi-task BCE loss with masking
                        raw_loss = criterion(outputs, labels)  # [batch_size, num_tasks]
                        masked_loss = raw_loss * masks.float()

                        valid_loss_sum = masked_loss.sum()
                        valid_count = masks.float().sum()

                        if valid_count > 0:
                            loss = valid_loss_sum / valid_count
                        else:
                            loss = torch.tensor(0.0, device=self.device)

                        # Multi-task accuracy calculation
                        sigmoid_outputs = torch.sigmoid(outputs)
                        predicted = (sigmoid_outputs > 0.5).float()
                        correct = (predicted == labels).float() * masks.float()
                        val_correct += correct.sum().item()
                        val_total += masks.float().sum().item()
                    else:
                        # Single task - classification or regression
                        if self.is_regression:
                            # Regression task
                            if len(labels.shape) > 1:
                                labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.view(-1)
                            if labels.dtype != torch.float:
                                labels = labels.float()
                            outputs = outputs.squeeze(-1) if outputs.size(-1) == 1 else outputs
                            loss = criterion(outputs, labels)

                            # For regression, we use MAE as "accuracy" metric
                            mae = torch.abs(outputs - labels).mean()
                            val_correct += mae.item() * labels.size(0)  # Store total MAE for averaging
                            val_total += labels.size(0)
                        else:
                            # Classification task
                            if len(labels.shape) > 1:
                                labels = labels.squeeze(-1) if labels.size(-1) == 1 else labels.argmax(dim=1)
                            if labels.dtype != torch.long:
                                labels = labels.long()
                            loss = criterion(outputs, labels)

                            pred = outputs.argmax(dim=1)
                            val_correct += (pred == labels).sum().item()
                            val_total += labels.size(0)

                    val_loss += loss.item()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # Calculate timing statistics
            avg_epoch_time_so_far = sum(epoch_times) / len(epoch_times)
            elapsed_time = time.time() - train_start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_mins = int((elapsed_time % 3600) // 60)
            elapsed_secs = int(elapsed_time % 60)

            # Estimate remaining time
            epochs_remaining = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time_so_far * epochs_remaining
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)

            if use_dam:
                # For DAM, use AUROC instead of accuracy
                val_auroc = _eval_val_auroc(val_loader)
                logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/len(train_loader):.4f}, "
                           f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
                           f"Val AUROC: {val_auroc:.4f}")
                logger.info(f"  Time: {epoch_time:.2f}s | Avg: {avg_epoch_time_so_far:.2f}s | Elapsed: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | ETA: {eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}")
                better_val = val_auroc > best_val_auc
                current_val_metric = val_auroc
                best_val_metric = best_val_auc
                metric_name = 'val_auc'
            else:
                val_acc = val_correct / val_total

                if self.is_regression:
                    logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/len(train_loader):.4f}, "
                               f"Train MAE: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
                               f"Val MAE: {val_acc:.4f}")
                    logger.info(f"  Time: {epoch_time:.2f}s | Avg: {avg_epoch_time_so_far:.2f}s | Elapsed: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | ETA: {eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}")
                    # For regression, lower MAE is better
                    if val_acc < best_val_acc or epoch == 0:
                        better_val = True
                    else:
                        better_val = False
                else:
                    logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss/len(train_loader):.4f}, "
                               f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
                               f"Val Acc: {val_acc:.4f}")
                    logger.info(f"  Time: {epoch_time:.2f}s | Avg: {avg_epoch_time_so_far:.2f}s | Elapsed: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | ETA: {eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}")
                    better_val = val_acc > best_val_acc
                current_val_metric = val_acc
                best_val_metric = best_val_acc
                metric_name = 'val_acc'

            # Early stopping and checkpointing
            if better_val:
                if use_dam:
                    best_val_auc = val_auroc
                else:
                    best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                os.makedirs(getattr(self.args, 'output_dir', './'), exist_ok=True)
                checkpoint_path = os.path.join(getattr(self.args, 'output_dir', './'), 'best_baseline.pth')
                checkpoint_data = {
                    'model_state_dict': self.model.state_dict(),
                    'args': vars(self.args),
                    'input_dim': self.input_dim,
                    'num_classes': self.num_classes,
                    'epoch': epoch,
                    'use_foundation_features': True
                }
                checkpoint_data[metric_name] = current_val_metric
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(f"Saved best model with {metric_name}: {current_val_metric:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Calculate timing and memory stats
        total_train_time = time.time() - train_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

        # Format total training time
        total_hours = int(total_train_time // 3600)
        total_mins = int((total_train_time % 3600) // 60)
        total_secs = int(total_train_time % 60)

        # Get peak GPU memory
        peak_gpu_memory_gb = 0
        if torch.cuda.is_available():
            peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        final_checkpoint = os.path.join(getattr(self.args, 'output_dir', './'), 'best_baseline.pth')
        if use_dam:
            logger.info(f"Training completed. Best validation AUROC: {best_val_auc:.4f}")
        else:
            logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"Model saved to: {final_checkpoint}")
        logger.info(f"Total training time: {total_hours:02d}:{total_mins:02d}:{total_secs:02d} ({total_train_time:.2f} seconds)")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        logger.info(f"Peak GPU memory (training): {peak_gpu_memory_gb:.2f}GB")
        if peak_encoding_memory_gb > 0:
            logger.info(f"Peak GPU memory (encoding): {peak_encoding_memory_gb:.2f}GB")

        # Store training features for KNN and LOF methods
        # Load the best model first to ensure we use the best weights
        best_model = self.load_trained_model(final_checkpoint)

        # Store training features - this will be used by KNN/LOF methods later
        self.best_trained_model = best_model

        # Return checkpoint path and timing/memory stats (including encoding memory)
        return {
            'checkpoint': final_checkpoint,
            'train_time_seconds': total_train_time,
            'avg_epoch_time_seconds': avg_epoch_time,
            'peak_gpu_memory_train_gb': peak_gpu_memory_gb,
            'peak_gpu_memory_encoding_gb': peak_encoding_memory_gb
        }

    def load_trained_model(self, checkpoint_path):
        """Load the trained model from checkpoint."""
        logger.info(f"Loading trained model from {checkpoint_path}")

        # Recreate the model with same architecture
        if hasattr(self, 'is_multitask') and self.is_multitask:
            # Multi-task model
            model = GCN_Classifier(
                input_dim=self.input_dim,
                hidden_dim=getattr(self.args, 'hidden_channels', 64),
                output_dim=self.num_classes,
                num_layers=getattr(self.args, 'num_layers', 3),
                dropout=getattr(self.args, 'dropout', 0.5),
                multitask=True
            )
        else:
            # Single task model (classification or regression)
            model = GCN_Classifier(
                input_dim=self.input_dim,
                hidden_dim=getattr(self.args, 'hidden_channels', 64),
                output_dim=self.num_classes,
                num_layers=getattr(self.args, 'num_layers', 3),
                dropout=getattr(self.args, 'dropout', 0.5)
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        logger.info("Model loaded successfully")
        return model


class BaselineEvaluator:
    """
    Evaluates trained baseline models using SupervisedBaselineDataLoader test splits.
    """
    
    def __init__(self, checkpoint_path, method_name, args):
        self.checkpoint_path = checkpoint_path
        self.method_name = method_name.lower()
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SupervisedBaselineDataLoader (reuses same data splits and cache)
        self.data_loader = SupervisedBaselineDataLoader(args)
        
        # Load trained model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Store metadata for downstream logic
        self.num_classes = checkpoint.get('num_classes', None)

        # Recreate model architecture
        model = GCN_Classifier(
            input_dim=checkpoint['input_dim'],
            hidden_dim=getattr(self.args, 'hidden_channels', 64),
            output_dim=checkpoint['num_classes'],
            num_layers=getattr(self.args, 'num_layers', 3),
            dropout=getattr(self.args, 'dropout', 0.5),
            use_features=checkpoint.get('use_foundation_features', True)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully: val_acc={checkpoint.get('val_acc', 'N/A')}")
        return model
        
    def evaluate(self):
        """Evaluate OOD detection using proper test splits."""
        logger.info(f"Starting {self.method_name.upper()} baseline evaluation...")

        # Start timing and reset GPU memory (consistent with evaluation.py)
        eval_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Warn: MSP and Energy are equivalent (up to monotone transform) for 1-D outputs
        try:
            if self.method_name in ['msp', 'energy'] and getattr(self, 'num_classes', None) == 1:
                logger.warning(
                    "Detected 1-D regression head. MSP and Energy both reduce to monotone functions of |logit| "
                    "(1 - sigmoid(|z|) vs -log(2*cosh(|z|))). Rankings and AUROC/AUPR will be identical. "
                    "Consider using classification datasets (e.g., GOOD-HIV/PCBA) or feature-space methods (Mahalanobis/KNN/LOF)."
                )
        except Exception:
            pass
        
        # Get test data loaders from SupervisedBaselineDataLoader
        batch_size = getattr(self.args, 'eval_batch_size', 64)
        num_workers = min(getattr(self.args, 'num_workers', 4), 2)
        
        id_test_loader, ood_test_loader = self.data_loader.get_test_loaders(batch_size, num_workers)
        
        logger.info(f"Test data: ID={len(id_test_loader.dataset)} samples, "
                   f"OOD={len(ood_test_loader.dataset)} samples")
        
        # Map DAM methods to scoring methods
        score_method = self.method_name
        if score_method.startswith('dam_'):
            score_method = score_method.split('_', 1)[1]  # dam_msp -> msp, dam_energy -> energy

        # Create baseline OOD model
        baseline_model = BaselineOODModel(self.model, score_method, self.args)
        
        # For Mahalanobis method, fit statistics using training data
        if score_method == 'mahalanobis':
            logger.info("Fitting class statistics for Mahalanobis method...")
            self._fit_mahalanobis_statistics(baseline_model)

        # For KNN and LOF methods, store training features
        if score_method in ['knn', 'lof']:
            logger.info(f"Storing training features for {self.method_name.upper()} method...")
            self._store_training_features(baseline_model)

        # For Conformal Prediction method, fit calibration on validation data
        if score_method == 'conformal':
            logger.info("Fitting conformal calibration on validation set...")
            self._fit_conformal_calibration(baseline_model)

        # Compute OOD scores
        logger.info("Computing OOD scores...")
        id_scores = self._get_ood_scores(baseline_model, id_test_loader, "ID")
        ood_scores = self._get_ood_scores(baseline_model, ood_test_loader, "OOD")
        
        # Evaluate metrics
        results = self._compute_metrics(id_scores, ood_scores)

        # Calculate timing and memory stats (consistent with evaluation.py)
        eval_time = time.time() - eval_start_time

        # Format evaluation time
        eval_hours = int(eval_time // 3600)
        eval_mins = int((eval_time % 3600) // 60)
        eval_secs = int(eval_time % 60)

        peak_gpu_memory_gb = 0
        if torch.cuda.is_available():
            peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results['eval_time_seconds'] = eval_time
        results['peak_gpu_memory_eval_gb'] = peak_gpu_memory_gb

        logger.info(f"  Eval time: {eval_hours:02d}:{eval_mins:02d}:{eval_secs:02d} ({eval_time:.2f} seconds)")
        logger.info(f"  Peak GPU memory (evaluation): {peak_gpu_memory_gb:.2f}GB")

        # Save and print results
        self._save_results(results, id_scores, ood_scores)
        self._print_results(results)

        return results
    
    def _fit_mahalanobis_statistics(self, baseline_model):
        """Fit class statistics for Mahalanobis method using training data."""
        # Get training loader for statistics fitting
        batch_size = getattr(self.args, 'eval_batch_size', 64)
        num_workers = min(getattr(self.args, 'num_workers', 4), 2)
        
        train_loader = self.data_loader.get_training_loader(batch_size, num_workers)
        
        # Get number of classes from checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        num_classes = checkpoint.get('num_classes', 2)
        
        baseline_model.fit_class_statistics_from_dataloader(train_loader, self.device, num_classes)

    def _store_training_features(self, baseline_model):
        """Store training features for KNN and LOF methods."""
        # Get training loader
        batch_size = getattr(self.args, 'eval_batch_size', 64)
        num_workers = min(getattr(self.args, 'num_workers', 4), 2)

        train_loader = self.data_loader.get_training_loader(batch_size, num_workers)

        # Store training features in the baseline model
        baseline_model.store_training_features(train_loader, self.device)

    def _fit_conformal_calibration(self, baseline_model):
        """Fit conformal calibration using validation data."""
        # Get validation loader for calibration
        batch_size = getattr(self.args, 'eval_batch_size', 64)
        num_workers = min(getattr(self.args, 'num_workers', 4), 2)

        val_loader = self.data_loader.get_validation_loader(batch_size, num_workers)

        # Fit calibration scores in the baseline model
        baseline_model.fit_conformal_calibration(val_loader, self.device)

    def _get_ood_scores(self, baseline_model, data_loader, data_type):
        """Get OOD scores for all samples."""
        all_scores = []
        
        # IMPORTANT: For ODIN method, we need to set the model to training mode
        # to enable gradient computation, but other methods use eval mode
        if self.method_name == 'odin':
            baseline_model.train()  # Enable gradients for ODIN
        else:
            baseline_model.eval()
        
        with torch.no_grad() if self.method_name != 'odin' else torch.enable_grad():
            for batch in tqdm(data_loader, desc=f"Computing {self.method_name} scores ({data_type})"):
                # Handle different batch formats
                if isinstance(batch, dict) and 'features' in batch:
                    features = batch['features'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                    features = batch[0].to(self.device)  # TensorDataset format
                else:
                    features = batch.to(self.device)
                
                # Ensure features have the correct dtype and gradient settings
                features = features.float()
                
                scores = baseline_model.detect(features, self.device)
                all_scores.extend(scores.cpu().numpy())
        
        logger.info(f"{data_type} scores: mean={np.mean(all_scores):.4f}, std={np.std(all_scores):.4f}")
        return np.array(all_scores)
    
    def _compute_metrics(self, id_scores, ood_scores):
        """Compute OOD detection metrics."""
        # Calculate separation for logging
        separation = np.mean(ood_scores) - np.mean(id_scores)

        # Create labels (0 for ID, 1 for OOD)
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])

        # Compute metrics
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not compute AUROC: {e}")
            auroc = 0.5

        try:
            aupr = average_precision_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not compute AUPR: {e}")
            aupr = 0.5

        fpr95 = self._compute_fpr95(id_scores, ood_scores)

        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr95,
            'id_scores_mean': np.mean(id_scores),
            'id_scores_std': np.std(id_scores),
            'ood_scores_mean': np.mean(ood_scores),
            'ood_scores_std': np.std(ood_scores),
            'separation': separation
        }
    
    def _compute_fpr95(self, id_scores, ood_scores):
        """Compute FPR at 95% TPR."""
        try:
            ood_scores_sorted = np.sort(ood_scores)
            threshold_idx = int(0.05 * len(ood_scores))
            threshold = ood_scores_sorted[threshold_idx]
            
            false_positives = np.sum(id_scores >= threshold)
            fpr95 = false_positives / len(id_scores)
        except (IndexError, ZeroDivisionError):
            logger.warning("Could not compute FPR95, returning 1.0")
            fpr95 = 1.0
        
        return fpr95
    
    def _save_results(self, results, id_scores, ood_scores):
        """Save detailed results."""
        output_dir = getattr(self.args, 'output_dir', './')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary results
        results_file = os.path.join(output_dir, f'{self.method_name}_results.json')
        import json
        with open(results_file, 'w') as f:
            serializable_results = {k: float(v) for k, v in results.items()}
            json.dump(serializable_results, f, indent=2)
        
        # Save raw scores
        scores_file = os.path.join(output_dir, f'{self.method_name}_scores.npz')
        np.savez(scores_file, id_scores=id_scores, ood_scores=ood_scores)
        
        logger.info(f"Results saved to: {results_file}")
    
    def _print_results(self, results):
        """Print formatted results."""
        logger.info("\n" + "="*60)
        logger.info(f"{self.method_name.upper()} BASELINE RESULTS")
        logger.info("="*60)
        logger.info(f"AUROC: {results['auroc']:.4f}")
        logger.info(f"AUPR:  {results['aupr']:.4f}")
        logger.info(f"FPR95: {results['fpr95']:.4f}")
        logger.info("="*60)


def run_baseline_training(args):
    """Run baseline training."""
    trainer = BaselineTrainer(args)
    return trainer.train()


def run_baseline_evaluation(checkpoint_path, method_name, args):
    """Run baseline evaluation."""
    evaluator = BaselineEvaluator(checkpoint_path, method_name, args)
    return evaluator.evaluate()


def run_all_baselines(args):
    """Run all baseline methods."""
    logger.info("Starting comprehensive baseline evaluation...")
    
    # Train model once
    checkpoint_path = run_baseline_training(args)

    # Evaluate all methods
    methods = ['msp', 'energy', 'odin', 'mahalanobis', 'knn', 'lof', 'mc_dropout', 'conformal']
    all_results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method.upper()}...")
        try:
            results = run_baseline_evaluation(checkpoint_path, method, args)
            all_results[method] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {str(e)}")
            all_results[method] = None
    
    # Print summary
    logger.info("\nCOMPREHENSIVE RESULTS:")
    logger.info("-" * 50)
    for method, results in all_results.items():
        if results:
            logger.info(f"{method.upper()}: AUROC={results['auroc']:.4f}")
        else:
            logger.info(f"{method.upper()}: FAILED")
    
    return all_results
