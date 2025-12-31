import os
import json
import glob
import logging
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from data_loader import EnergyDPODataLoader
from model import create_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--dataset", type=str, default="drugood")
    parser.add_argument("--drugood_subset", type=str, default="lbap_general_ec50_scaffold")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--data_file", type=str, help="Specific data file path")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path for resuming training")

    # Training
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Others
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    return parser.parse_args()

class EnergyDPOTrainer:
    def __init__(self, args):

        self.args = args
        
        # 1. First set device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 2. Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # 3. Create model
        self.model = create_model(args).to(self.device)

        # 4. Create optimizer and scheduler
        # Check if encoder finetuning is enabled
        finetune_encoder = getattr(args, 'finetune_encoder', False)

        if finetune_encoder:
            # Separate parameter groups for encoder and head
            encoder_params = []
            head_params = []

            for name, param in self.model.named_parameters():
                if 'encoder' in name and param.requires_grad:
                    encoder_params.append(param)
                else:
                    head_params.append(param)

            # Different learning rates for encoder and head
            encoder_lr = getattr(args, 'encoder_lr', 5e-6)
            head_lr = args.lr

            self.optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
                {'params': head_params, 'lr': head_lr, 'name': 'head'}
            ], weight_decay=args.weight_decay)

            logger.info(f"Finetuning mode enabled:")
            logger.info(f"  Encoder params: {len(encoder_params)} @ LR={encoder_lr}")
            logger.info(f"  Head params: {len(head_params)} @ LR={head_lr}")

            # Adjust batch size for finetuning (to avoid OOM)
            if not hasattr(args, 'original_batch_size'):
                args.original_batch_size = args.batch_size
            if args.batch_size > 256:
                logger.info(f"Reducing batch size from {args.batch_size} to 256 for finetuning")
                args.batch_size = 256
            if args.eval_batch_size > 512:
                logger.info(f"Reducing eval batch size from {args.eval_batch_size} to 512 for finetuning")
                args.eval_batch_size = 512
        else:
            # Default: single optimizer for all parameters (frozen encoder)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        # Enable mixed precision on CUDA to reduce memory footprint
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )

        # 5. Initialize training state (set only once)
        self.start_epoch = 0
        self.global_step = 0
        self.best_eval_metric = float('-inf')
        self.patience = getattr(self.args, 'early_stopping_patience', 10)
        self.epochs_no_improve = 0

        # 6. Training dynamics tracking for "hard pairs corrected first" analysis
        self.training_dynamics = []

        # 6. Finally load checkpoint (if exists)
        self.load_checkpoint_if_exists()

        logger.info(f"Trainer initialization complete")
        logger.info(f"Starting epoch: {self.start_epoch + 1}")
        logger.info(f"Device: {self.device}")
    
    def load_checkpoint_if_exists(self):
        """Check and load checkpoint"""
        if hasattr(self.args, 'model_path') and self.args.model_path and os.path.exists(self.args.model_path):
            try:
                logger.info(f"Loading checkpoint: {self.args.model_path}")
                checkpoint = torch.load(self.args.model_path, map_location=self.device)
                
                # Load model state
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Model state loaded")
                
                # Load optimizer state  
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded")
                
                # Load scheduler state
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state loaded")
                
                # Load training state
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                    logger.info(f"Will continue training from epoch {self.start_epoch + 1}")
                
                if 'global_step' in checkpoint:
                    self.global_step = checkpoint['global_step']
                    logger.info(f"Global step: {self.global_step}")
                
                if 'best_eval_metric' in checkpoint:
                    self.best_eval_metric = checkpoint['best_eval_metric']
                    logger.info(f"Best metric: {self.best_eval_metric:.4f}")
                
                logger.info(f"Successfully restored training state from checkpoint!")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Will start training from scratch")
                self.start_epoch = 0
                self.global_step = 0
                self.best_eval_metric = float('-inf')
        else:
            if hasattr(self.args, 'model_path') and self.args.model_path:
                logger.warning(f"Checkpoint file does not exist: {self.args.model_path}")
            logger.info("Starting fresh training")
    
    def compute_energy_dpo_loss(self, id_smiles=None, ood_smiles=None, batch_data=None):
        """DPO loss computation with support for precomputed features"""
        if batch_data is not None:
            # Pass directly to model, model will automatically choose path
            return self.model(batch_data)
        else:
            # Backward compatibility
            batch_data = {'id_smiles': id_smiles, 'ood_smiles': ood_smiles}
            return self.model(batch_data)
    
    def evaluate(self, eval_dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        energy_separations = []
        all_id_scores = []
        all_ood_scores = []
        amp_enabled = getattr(self, 'scaler', None) is not None and self.scaler.is_enabled()

        with torch.no_grad():
            for batch_data in eval_dataloader:
                # Compute loss (for logging)
                with autocast(enabled=amp_enabled):
                    loss, loss_dict = self.compute_energy_dpo_loss(batch_data=batch_data)
                total_loss += loss.item()
                energy_separations.append(loss_dict['energy_separation'])

                # Compute scores for AUROC (energy-based, OOD should be higher than ID)
                try:
                    if 'id_features' in batch_data and 'ood_features' in batch_data:
                        with autocast(enabled=amp_enabled):
                            id_scores = self.model.predict_ood_score_from_features(batch_data['id_features'])
                            ood_scores = self.model.predict_ood_score_from_features(batch_data['ood_features'])
                    elif 'id_graphs' in batch_data and 'ood_graphs' in batch_data:
                        # Graphs cached: encode then score
                        with autocast(enabled=amp_enabled):
                            id_feats = self.model.encoder.encode_smiles(batch_data['id_graphs'])
                            ood_feats = self.model.encoder.encode_smiles(batch_data['ood_graphs'])
                        id_scores = self.model.predict_ood_score_from_features(id_feats)
                        ood_scores = self.model.predict_ood_score_from_features(ood_feats)
                    else:
                        # Raw SMILES fallback
                        id_scores = self.model.predict_ood_score(batch_data['id_smiles'])
                        ood_scores = self.model.predict_ood_score(batch_data['ood_smiles'])
                except Exception:
                    # Last-resort fallback: compute directly through energy head if tensors are present
                    with autocast(enabled=amp_enabled):
                        id_tensor = batch_data.get('id_features') or self.model.encoder.encode_smiles(batch_data['id_smiles'])
                        ood_tensor = batch_data.get('ood_features') or self.model.encoder.encode_smiles(batch_data['ood_smiles'])
                        device = next(self.model.parameters()).device
                        id_scores = self.model.energy_head(id_tensor.to(device)).squeeze(-1).detach().cpu().numpy()
                        ood_scores = self.model.energy_head(ood_tensor.to(device)).squeeze(-1).detach().cpu().numpy()

                all_id_scores.append(id_scores)
                all_ood_scores.append(ood_scores)
        
        avg_loss = total_loss / max(1, len(eval_dataloader))
        avg_energy_sep = sum(energy_separations) / max(1, len(energy_separations))

        # Concatenate and compute AUROC / AUPR / FPR95
        import numpy as np
        id_scores_np = np.concatenate(all_id_scores) if len(all_id_scores) > 0 else np.array([])
        ood_scores_np = np.concatenate(all_ood_scores) if len(all_ood_scores) > 0 else np.array([])

        if id_scores_np.size > 0 and ood_scores_np.size > 0:
            labels = np.concatenate([np.zeros_like(id_scores_np), np.ones_like(ood_scores_np)])
            scores = np.concatenate([id_scores_np, ood_scores_np])
            try:
                val_auroc = roc_auc_score(labels, scores)
                precision, recall, _ = precision_recall_curve(labels, scores)
                val_aupr = auc(recall, precision)
                fpr, tpr, _ = roc_curve(labels, scores)
                idx = np.where(tpr >= 0.95)[0]
                val_fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0
            except Exception:
                val_auroc, val_aupr, val_fpr95 = 0.0, 0.0, 1.0
        else:
            val_auroc, val_aupr, val_fpr95 = 0.0, 0.0, 1.0
        
        return {
            'total_loss': avg_loss,
            'energy_separation': avg_energy_sep,
            'val_auroc': val_auroc,
            'val_aupr': val_aupr,
            'val_fpr95': val_fpr95,
        }
    
    def save_training_dynamics(self, epoch_data):
        """Save training dynamics data for "hard pairs corrected first" analysis"""
        import csv
        dynamics_path = os.path.join(self.args.output_dir, 'training_dynamics.csv')

        # Create CSV file with header if it doesn't exist
        file_exists = os.path.exists(dynamics_path)
        with open(dynamics_path, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'misranked_ratio', 'boundary_ratio', 'avg_margin']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(epoch_data)

        logger.info(f"Training dynamics saved: epoch {epoch_data['epoch']}, "
                   f"misranked: {epoch_data['misranked_ratio']:.4f}, "
                   f"boundary: {epoch_data['boundary_ratio']:.4f}, "
                   f"margin: {epoch_data['avg_margin']:.4f}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_metric': self.best_eval_metric,
            'args': self.args
        }
        
        # Save best model
        if is_best:
            filename = 'best_model.pth'
            save_path = os.path.join(self.args.output_dir, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved to {save_path}")
        
        # Periodically save epoch checkpoints
        if epoch % 5 == 0 or epoch == self.args.epochs - 1:
            filename = f'checkpoint_epoch_{epoch:03d}.pth'
            save_path = os.path.join(self.args.output_dir, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
            
            # Clean up old files, keep only the latest 3 epoch files
            pattern = os.path.join(self.args.output_dir, 'checkpoint_epoch_*.pth')
            old_files = sorted(glob.glob(pattern))
            if len(old_files) > 3:
                for old_file in old_files[:-3]:
                    try:
                        os.remove(old_file)
                        logger.info(f"Cleaned up old file: {os.path.basename(old_file)}")
                    except:
                        pass
    
    def train(self):
        """Start training"""
        # Start timing
        train_start_time = time.time()
        epoch_times = []

        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Starting epoch: {self.start_epoch + 1}")
        logger.info(f"Total epochs: {self.args.epochs}")

        # Load data (encoding happens here)
        data_loader = EnergyDPODataLoader(self.args)

        # Get encoding memory if available
        peak_encoding_memory_gb = getattr(data_loader, 'peak_encoding_memory_gb', 0.0)
        if peak_encoding_memory_gb > 0:
            logger.info(f"Foundation model encoding peak GPU memory: {peak_encoding_memory_gb:.2f} GB")

        # Reset GPU memory stats for training phase
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_dataloader, eval_dataloader = data_loader.get_dataloaders()
        
        # Training loop - Use self.start_epoch as starting point
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start_time = time.time()

            self.model.train()
            train_losses = []
            # Training dynamics data collection
            epoch_dynamics = {'misranked_ratio': [], 'boundary_ratio': [], 'avg_margin': []}

            # Set up progress bar
            use_tqdm = os.getenv('TQDM_DISABLE', '0') == '0'
            if use_tqdm:
                progress_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                progress_iter = train_dataloader
                total_batches = len(train_dataloader)
            
            for batch_idx, batch_data in enumerate(progress_iter):
                # Pass batch_data directly, model will automatically choose optimal path
                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    loss, loss_dict = self.compute_energy_dpo_loss(batch_data=batch_data)
                
                # Backpropagation with optional AMP
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()
                self.global_step += 1
                
                # Record loss
                train_losses.append(loss_dict)

                # Collect training dynamics data if available
                if 'misranked_ratio' in loss_dict:
                    epoch_dynamics['misranked_ratio'].append(loss_dict['misranked_ratio'])
                    epoch_dynamics['boundary_ratio'].append(loss_dict['boundary_ratio'])
                    epoch_dynamics['avg_margin'].append(loss_dict['avg_margin'])
                
                # Update progress display
                if use_tqdm:
                    progress_iter.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'E_Sep': f"{loss_dict['energy_separation']:.4f}"
                    })
                else:
                    # Simple progress display
                    progress = (batch_idx + 1) / total_batches * 100
                    bar_length = 30
                    filled_length = int(bar_length * (batch_idx + 1) // total_batches)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\rEpoch {epoch+1}/{self.args.epochs} |{bar}| {progress:5.1f}% ({batch_idx+1}/{total_batches}) "
                          f"Loss: {loss.item():.4f} E_Sep: {loss_dict['energy_separation']:.4f}", end='')
                
                # Periodic evaluation
                if self.global_step % self.args.eval_steps == 0:
                    eval_msg = f"Step {self.global_step} - Starting evaluation..."
                    if use_tqdm:
                        progress_iter.write(eval_msg)
                    else:
                        print(f"\n{eval_msg}")
                        logger.info(eval_msg)
                    
                    eval_loss_dict = self.evaluate(eval_dataloader)
                    
                    result_msg = (
                        f"Step {self.global_step} - Validation loss: {eval_loss_dict['total_loss']:.4f}, "
                        f"Energy separation: {eval_loss_dict['energy_separation']:.4f}, "
                        f"Val-AUROC: {eval_loss_dict['val_auroc']:.4f}"
                    )
                    if use_tqdm:
                        progress_iter.write(result_msg)
                    else:
                        print(result_msg)
                        logger.info(result_msg)
                    
                    # Check if this is the best model
                    current_metric = eval_loss_dict['val_auroc']
                    if current_metric > self.best_eval_metric:
                        self.best_eval_metric = current_metric
                        self.save_checkpoint(epoch, is_best=True)
                        best_msg = f"Found better model! Val-AUROC: {current_metric:.4f}"
                        if use_tqdm:
                            progress_iter.write(best_msg)
                        else:
                            print(best_msg)
                            logger.info(best_msg)
            
            # Complete progress display for current epoch
            if not use_tqdm:
                print()  # Ensure newline
            elif use_tqdm:
                progress_iter.close()
            
            # Evaluation at epoch end (based on Val-AUROC selection)
            eval_loss_dict = self.evaluate(eval_dataloader)
            
            # Calculate average training loss
            avg_train_loss = sum(loss['total_loss'] for loss in train_losses) / len(train_losses)

            # Save training dynamics data at epoch end
            if epoch_dynamics['misranked_ratio']:  # Only if we have dynamics data
                avg_dynamics = {
                    'epoch': epoch + 1,
                    'misranked_ratio': sum(epoch_dynamics['misranked_ratio']) / len(epoch_dynamics['misranked_ratio']),
                    'boundary_ratio': sum(epoch_dynamics['boundary_ratio']) / len(epoch_dynamics['boundary_ratio']),
                    'avg_margin': sum(epoch_dynamics['avg_margin']) / len(epoch_dynamics['avg_margin'])
                }
                self.save_training_dynamics(avg_dynamics)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # Calculate timing statistics
            avg_epoch_time_so_far = sum(epoch_times) / len(epoch_times)
            elapsed_time = time.time() - train_start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_mins = int((elapsed_time % 3600) // 60)
            elapsed_secs = int(elapsed_time % 60)

            # Estimate remaining time
            epochs_completed = len(epoch_times)
            epochs_remaining = self.args.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time_so_far * epochs_remaining
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)

            logger.info(f"Epoch {epoch+1} completed:")
            logger.info(f"  Training loss: {avg_train_loss:.4f}")
            logger.info(f"  Validation loss: {eval_loss_dict['total_loss']:.4f}")
            logger.info(f"  Energy separation: {eval_loss_dict['energy_separation']:.4f}")
            logger.info(f"  Val-AUROC: {eval_loss_dict['val_auroc']:.4f} (AUPR: {eval_loss_dict['val_aupr']:.4f}, FPR95: {eval_loss_dict['val_fpr95']:.4f})")
            logger.info(f"  Epoch time: {epoch_time:.2f}s | Avg: {avg_epoch_time_so_far:.2f}s | Elapsed: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | ETA: {eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}")

            # Update learning rate
            self.scheduler.step()
            
            # Early stopping and best model saving based on Val-AUROC
            current_metric = eval_loss_dict['val_auroc']
            if current_metric > self.best_eval_metric:
                self.best_eval_metric = current_metric
                self.save_checkpoint(epoch, is_best=True)
                self.epochs_no_improve = 0
                logger.info(f"Found better model in epoch! Val-AUROC: {current_metric:.4f}")
            
            # Save checkpoint for this epoch
            self.save_checkpoint(epoch)

            # Early stopping check
            if current_metric <= self.best_eval_metric + 1e-12:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered: {self.epochs_no_improve} consecutive epochs without improvement (patience={self.patience})")
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

        logger.info("Training completed!")
        logger.info(f"Best Val-AUROC: {self.best_eval_metric:.4f}")
        logger.info(f"Total training time: {total_hours:02d}:{total_mins:02d}:{total_secs:02d} ({total_train_time:.2f} seconds)")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        logger.info(f"Peak GPU memory (training): {peak_gpu_memory_gb:.2f}GB")
        if peak_encoding_memory_gb > 0:
            logger.info(f"Peak GPU memory (encoding): {peak_encoding_memory_gb:.2f}GB")

        # Save configuration file
        args_path = os.path.join(self.args.output_dir, 'config.json')
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)

        # Return timing and memory stats (including encoding memory)
        return {
            'train_time_seconds': total_train_time,
            'avg_epoch_time_seconds': avg_epoch_time,
            'peak_gpu_memory_train_gb': peak_gpu_memory_gb,
            'peak_gpu_memory_encoding_gb': peak_encoding_memory_gb
        }

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
    trainer = EnergyDPOTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
