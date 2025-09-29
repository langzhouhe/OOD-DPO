import os
import json
import glob
import logging
import torch
import torch.optim as optim
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
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
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
                
                # 加载模型状态
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Model state loaded")
                
                # 加载优化器状态  
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ 优化器状态已加载")
                
                # 加载调度器状态
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("✅ 调度器状态已加载")
                
                # 加载训练状态
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                    logger.info(f"✅ 将从第 {self.start_epoch + 1} 个epoch继续训练")
                
                if 'global_step' in checkpoint:
                    self.global_step = checkpoint['global_step']
                    logger.info(f"✅ Global step: {self.global_step}")
                
                if 'best_eval_metric' in checkpoint:
                    self.best_eval_metric = checkpoint['best_eval_metric']
                    logger.info(f"✅ 最佳指标: {self.best_eval_metric:.4f}")
                
                logger.info(f"🎉 成功从checkpoint恢复训练状态！")
                
            except Exception as e:
                logger.error(f"❌ 加载checkpoint失败: {e}")
                logger.info("🔄 将从头开始训练")
                self.start_epoch = 0
                self.global_step = 0
                self.best_eval_metric = float('-inf')
        else:
            if hasattr(self.args, 'model_path') and self.args.model_path:
                logger.warning(f"⚠️ Checkpoint文件不存在: {self.args.model_path}")
            logger.info("🚀 开始全新训练")
    
    def compute_energy_dpo_loss(self, id_smiles=None, ood_smiles=None, batch_data=None):
        """支持预计算特征的DPO损失计算"""
        if batch_data is not None:
            # 直接传递给模型，模型会自动选择路径
            return self.model(batch_data)
        else:
            # 向后兼容
            batch_data = {'id_smiles': id_smiles, 'ood_smiles': ood_smiles}
            return self.model(batch_data)
    
    def evaluate(self, eval_dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        energy_separations = []
        all_id_scores = []
        all_ood_scores = []
        
        with torch.no_grad():
            for batch_data in eval_dataloader:
                # 计算损失（用于日志）
                loss, loss_dict = self.compute_energy_dpo_loss(batch_data=batch_data)
                total_loss += loss.item()
                energy_separations.append(loss_dict['energy_separation'])

                # 计算分数用于 AUROC（基于能量，OOD 应高于 ID）
                try:
                    id_scores = self.model.predict_ood_score_from_features(batch_data['id_features'])
                    ood_scores = self.model.predict_ood_score_from_features(batch_data['ood_features'])
                except Exception:
                    # 回退: 直接通过能量头计算
                    id_scores = self.model.energy_head(batch_data['id_features'].to(next(self.model.parameters()).device)).squeeze(-1).detach().cpu().numpy()
                    ood_scores = self.model.energy_head(batch_data['ood_features'].to(next(self.model.parameters()).device)).squeeze(-1).detach().cpu().numpy()

                all_id_scores.append(id_scores)
                all_ood_scores.append(ood_scores)
        
        avg_loss = total_loss / max(1, len(eval_dataloader))
        avg_energy_sep = sum(energy_separations) / max(1, len(energy_separations))

        # 拼接并计算 AUROC / AUPR / FPR95
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
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_metric': self.best_eval_metric,
            'args': self.args
        }
        
        # 保存最佳模型
        if is_best:
            filename = 'best_model.pth'
            save_path = os.path.join(self.args.output_dir, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"✅ 最佳模型已保存到 {save_path}")
        
        # 定期保存epoch检查点
        if epoch % 5 == 0 or epoch == self.args.epochs - 1:
            filename = f'checkpoint_epoch_{epoch:03d}.pth'
            save_path = os.path.join(self.args.output_dir, filename)
            torch.save(checkpoint, save_path)
            logger.info(f"检查点已保存到 {save_path}")
            
            # 清理旧文件，只保留最近3个epoch文件
            pattern = os.path.join(self.args.output_dir, 'checkpoint_epoch_*.pth')
            old_files = sorted(glob.glob(pattern))
            if len(old_files) > 3:
                for old_file in old_files[:-3]:
                    try:
                        os.remove(old_file)
                        logger.info(f"清理旧文件: {os.path.basename(old_file)}")
                    except:
                        pass
    
    def train(self):
        """Start training"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"输出目录: {self.args.output_dir}")
        logger.info(f"起始epoch: {self.start_epoch + 1}")
        logger.info(f"总epochs: {self.args.epochs}")
        
        # 加载数据
        data_loader = EnergyDPODataLoader(self.args)
        train_dataloader, eval_dataloader = data_loader.get_dataloaders()
        
        # 训练循环 - 🔥 使用self.start_epoch作为起始点
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            train_losses = []
            # Training dynamics data collection
            epoch_dynamics = {'misranked_ratio': [], 'boundary_ratio': [], 'avg_margin': []}
            
            # 设置进度条
            use_tqdm = os.getenv('TQDM_DISABLE', '0') == '0'
            if use_tqdm:
                progress_iter = tqdm(train_dataloader, desc=f"🚀 Epoch {epoch+1}/{self.args.epochs}")
            else:
                progress_iter = train_dataloader
                total_batches = len(train_dataloader)
            
            for batch_idx, batch_data in enumerate(progress_iter):
                # 直接传递batch_data，模型会自动选择最优路径
                loss, loss_dict = self.compute_energy_dpo_loss(batch_data=batch_data)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                self.global_step += 1
                
                # 记录损失
                train_losses.append(loss_dict)

                # Collect training dynamics data if available
                if 'misranked_ratio' in loss_dict:
                    epoch_dynamics['misranked_ratio'].append(loss_dict['misranked_ratio'])
                    epoch_dynamics['boundary_ratio'].append(loss_dict['boundary_ratio'])
                    epoch_dynamics['avg_margin'].append(loss_dict['avg_margin'])
                
                # 更新进度显示
                if use_tqdm:
                    progress_iter.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'E_Sep': f"{loss_dict['energy_separation']:.4f}"
                    })
                else:
                    # 简单的进度显示
                    progress = (batch_idx + 1) / total_batches * 100
                    bar_length = 30
                    filled_length = int(bar_length * (batch_idx + 1) // total_batches)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r🚀 Epoch {epoch+1}/{self.args.epochs} |{bar}| {progress:5.1f}% ({batch_idx+1}/{total_batches}) "
                          f"Loss: {loss.item():.4f} E_Sep: {loss_dict['energy_separation']:.4f}", end='')
                
                # 定期评估
                if self.global_step % self.args.eval_steps == 0:
                    eval_msg = f"🔍 步骤 {self.global_step} - 开始评估..."
                    if use_tqdm:
                        progress_iter.write(eval_msg)
                    else:
                        print(f"\n{eval_msg}")
                        logger.info(eval_msg)
                    
                    eval_loss_dict = self.evaluate(eval_dataloader)
                    
                    result_msg = (
                        f"✅ 步骤 {self.global_step} - 验证损失: {eval_loss_dict['total_loss']:.4f}, "
                        f"能量分离: {eval_loss_dict['energy_separation']:.4f}, "
                        f"Val-AUROC: {eval_loss_dict['val_auroc']:.4f}"
                    )
                    if use_tqdm:
                        progress_iter.write(result_msg)
                    else:
                        print(result_msg)
                        logger.info(result_msg)
                    
                    # 检查是否是最佳模型
                    current_metric = eval_loss_dict['val_auroc']
                    if current_metric > self.best_eval_metric:
                        self.best_eval_metric = current_metric
                        self.save_checkpoint(epoch, is_best=True)
                        best_msg = f"🎉 发现更好的模型！Val-AUROC: {current_metric:.4f}"
                        if use_tqdm:
                            progress_iter.write(best_msg)
                        else:
                            print(best_msg)
                            logger.info(best_msg)
            
            # 完成当前epoch的进度显示
            if not use_tqdm:
                print()  # Ensure newline
            elif use_tqdm:
                progress_iter.close()
            
            # epoch结束时的评估（基于 Val-AUROC 选择）
            eval_loss_dict = self.evaluate(eval_dataloader)
            
            # 计算平均训练损失
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
            
            logger.info(f"Epoch {epoch+1} 完成:")
            logger.info(f"  训练损失: {avg_train_loss:.4f}")
            logger.info(f"  验证损失: {eval_loss_dict['total_loss']:.4f}")
            logger.info(f"  能量分离: {eval_loss_dict['energy_separation']:.4f}")
            logger.info(f"  Val-AUROC: {eval_loss_dict['val_auroc']:.4f} (AUPR: {eval_loss_dict['val_aupr']:.4f}, FPR95: {eval_loss_dict['val_fpr95']:.4f})")
            
            # 更新学习率
            self.scheduler.step()
            
            # 基于 Val-AUROC 的早停与最佳模型保存
            current_metric = eval_loss_dict['val_auroc']
            if current_metric > self.best_eval_metric:
                self.best_eval_metric = current_metric
                self.save_checkpoint(epoch, is_best=True)
                self.epochs_no_improve = 0
                logger.info(f"🎉 epoch 发现更好模型! Val-AUROC: {current_metric:.4f}")
            
            # 保存本epoch的检查点
            self.save_checkpoint(epoch)

            # 早停判断
            if current_metric <= self.best_eval_metric + 1e-12:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                logger.info(f"⏹️ 早停触发: 连续 {self.epochs_no_improve} 个epoch无提升 (patience={self.patience})")
                break
        
        logger.info("训练完成！")
        logger.info(f"最佳 Val-AUROC: {self.best_eval_metric:.4f}")
        
        # 保存配置文件
        args_path = os.path.join(self.args.output_dir, 'config.json')
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
    trainer = EnergyDPOTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
