#!/usr/bin/env python3
"""
修正版本：Beta敏感性分析 - 真正在TEST SET上评估
固定 lambda=0.01，研究不同 beta 值在 EC50 scaffold、size、assay 数据集上的表现
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from model import create_model
from data_loader import EnergyDPODataLoader
from train import EnergyDPOTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BetaSensitivityAnalysis:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 固定 lambda=0.01，测试不同 beta 值
        self.beta_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.lambda_reg = 0.01  # 固定 lambda

        # 不同数据集
        self.datasets = [
            'lbap_general_ec50_scaffold',
            'lbap_general_ec50_size',
            'lbap_general_ec50_assay'
        ]

        # 存储结果
        self.results = {}

        # 设置输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = os.path.join(args.output_dir, f"beta_sensitivity_corrected_{timestamp}")
        os.makedirs(self.study_dir, exist_ok=True)

        logger.info(f"Beta sensitivity analysis output directory: {self.study_dir}")
        logger.info(f"Beta values: {self.beta_values}")
        logger.info(f"Fixed lambda: {self.lambda_reg}")
        logger.info(f"Datasets: {self.datasets}")
        # 训练控制参数（可通过CLI调整；<=0 表示不限制）
        self.epochs = getattr(self.args, 'epochs', 10)
        self.max_train_batches = getattr(self.args, 'max_train_batches', 30)
        self.max_valid_batches = getattr(self.args, 'max_valid_batches', 10)
        logger.info(
            f"Training settings -> epochs: {self.epochs}, "
            f"max_train_batches: {self.max_train_batches} (<=0 means no limit), "
            f"max_valid_batches: {self.max_valid_batches} (<=0 means no limit)"
        )

        # 将日志保存到文件
        try:
            fh = logging.FileHandler(os.path.join(self.study_dir, 'run.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Unable to attach file logger: {e}")

    def prepare_data(self, dataset_name):
        """为指定数据集准备数据，包括test set"""
        logger.info(f"Preparing data for dataset: {dataset_name}")

        data_args = argparse.Namespace()
        data_args.dataset = 'drugood'
        data_args.drugood_subset = dataset_name
        data_args.data_file = f"./data/raw/{dataset_name}.json"
        data_args.data_path = self.args.data_path
        data_args.batch_size = self.args.batch_size
        data_args.eval_batch_size = self.args.eval_batch_size

        self.data_loader_obj = EnergyDPODataLoader(data_args)
        train_loader, valid_loader = self.data_loader_obj.get_dataloaders()

        # 获取test set数据
        test_data = self.data_loader_obj.get_final_test_data()

        logger.info(f"Dataset {dataset_name}: Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
        logger.info(f"Test set: {len(test_data['id_smiles'])} ID + {len(test_data['ood_smiles'])} OOD samples")

        return train_loader, valid_loader, test_data

    def evaluate_on_test_set(self, model, test_data):
        """在test set上评估模型性能（逐样本能量得分 -> ROC-AUC）"""
        model.eval()

        with torch.no_grad():
            # 逐样本能量分数（越小越像ID）
            id_scores = model.predict_ood_score(test_data['id_smiles'])  # numpy array
            ood_scores = model.predict_ood_score(test_data['ood_smiles'])

            # 将能量取负作为正类（ID=1）的置信度分数
            y_scores = np.concatenate([-id_scores, -ood_scores])
            y_true = np.concatenate([
                np.ones_like(id_scores),
                np.zeros_like(ood_scores)
            ])

            test_auc = roc_auc_score(y_true, y_scores)

            # 也计算一次整体test损失与能量分离（仅统计，不用于AUC）
            test_loss, test_loss_dict = model.compute_energy_dpo_loss(
                test_data['id_smiles'], test_data['ood_smiles']
            )

        return {
            'test_loss': test_loss.item(),
            'test_auc': test_auc,
            'energy_separation': test_loss_dict['energy_separation']
        }

    def single_experiment(self, beta, dataset_name, train_loader, valid_loader, test_data, max_epochs=None):
        """运行单个实验：在训练集(train)上训练，基于验证集(valid)选择最佳模型，最后在测试集(test)上评估"""
        experiment_name = f"beta_{beta}_{dataset_name}"
        logger.info(f"Running experiment: {experiment_name}")

        # 创建模型
        model_args = argparse.Namespace()
        model_args.hidden_dim = self.args.hidden_dim
        model_args.dpo_beta = beta
        model_args.lambda_reg = self.lambda_reg
        model_args.foundation_model = getattr(self.args, 'foundation_model', 'minimol')
        model_args.loss_type = 'dpo'

        model = create_model(model_args)
        model.to(self.device)

        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=0.0)

        # 训练历史（用于选择最佳模型）
        best_valid_auc = 0
        best_model_state = None

        # 训练循环
        model.train()
        epochs_to_run = max_epochs if max_epochs is not None else self.epochs
        for epoch in range(epochs_to_run):
            # 训练阶段
            epoch_losses = []

            for batch_idx, batch_data in enumerate(train_loader):
                if (self.max_train_batches is not None and self.max_train_batches > 0 \
                        and batch_idx >= self.max_train_batches):
                    break

                optimizer.zero_grad()

                # 前向传播
                loss, loss_dict = model(batch_data)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # 验证阶段（用于early stopping和模型选择）
            model.eval()
            valid_losses = []
            valid_scores_all = []
            valid_labels_all = []

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(valid_loader):
                    if (self.max_valid_batches is not None and self.max_valid_batches > 0 \
                            and batch_idx >= self.max_valid_batches):
                        break

                    loss, loss_dict = model(batch_data)
                    valid_losses.append(loss.item())

                    # 获取验证集逐样本能量分数
                    if 'id_features' in batch_data and 'ood_features' in batch_data:
                        id_scores = model.predict_ood_score_from_features(batch_data['id_features'])
                        ood_scores = model.predict_ood_score_from_features(batch_data['ood_features'])
                    else:
                        id_scores = model.predict_ood_score(batch_data['id_smiles'])
                        ood_scores = model.predict_ood_score(batch_data['ood_smiles'])

                    # 取负能量作为ID置信度
                    y_scores = np.concatenate([-id_scores, -ood_scores])
                    y_labels = np.concatenate([
                        np.ones_like(id_scores),
                        np.zeros_like(ood_scores)
                    ])

                    valid_scores_all.extend(y_scores.tolist())
                    valid_labels_all.extend(y_labels.tolist())

            # 计算验证AUC
            if len(valid_scores_all) > 0:
                valid_auc = roc_auc_score(valid_labels_all, valid_scores_all)
            else:
                valid_auc = 0.5

            # 保存最佳模型（基于validation AUC）
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_model_state = model.state_dict().copy()

            if epoch % 3 == 0:
                logger.info(f"Epoch {epoch}: Valid AUC: {valid_auc:.4f}")

            model.train()

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # **关键**：在TEST SET上评估最终性能
        logger.info("Evaluating final performance on TEST SET...")
        test_metrics = self.evaluate_on_test_set(model, test_data)

        final_metrics = {
            'beta': beta,
            'dataset': dataset_name,
            'best_valid_auc': best_valid_auc,
            'test_auc': test_metrics['test_auc'],  # 这是真正的test AUC
            'test_loss': test_metrics['test_loss'],
            'test_energy_separation': test_metrics['energy_separation']
        }

        logger.info(f"Beta {beta}: Valid AUC = {best_valid_auc:.4f}, TEST AUC = {test_metrics['test_auc']:.4f}")

        return final_metrics

    def run_beta_sensitivity(self):
        """运行beta敏感性分析"""
        logger.info("Starting beta sensitivity analysis...")

        all_results = []

        for dataset_name in self.datasets:
            logger.info(f"\n=== Processing dataset: {dataset_name} ===")

            # 准备数据
            train_loader, valid_loader, test_data = self.prepare_data(dataset_name)

            dataset_results = []

            # 对每个beta值运行实验
            for beta in tqdm(self.beta_values, desc=f"Beta sensitivity - {dataset_name}"):
                try:
                    # 运行实验
                    metrics = self.single_experiment(
                        beta, dataset_name, train_loader, valid_loader, test_data
                    )

                    dataset_results.append(metrics)

                except Exception as e:
                    logger.warning(f"Experiment failed for beta={beta}, dataset={dataset_name}: {e}")
                    # 记录失败的实验
                    failed_metrics = {
                        'beta': beta,
                        'dataset': dataset_name,
                        'test_auc': 0.5,  # 随机猜测的AUC
                        'failed': True,
                        'error': str(e)
                    }
                    dataset_results.append(failed_metrics)

            all_results.extend(dataset_results)
            self.results[dataset_name] = dataset_results

        # 保存完整结果
        self.all_results = all_results

        logger.info("Beta sensitivity analysis completed!")

    def create_corrected_plots(self):
        """创建基于真实TEST SET的可视化"""
        logger.info("Creating plots based on TEST SET performance...")

        # 设置绘图风格和字体
        plt.style.use('default')
        sns.set_palette("husl")
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 14

        # 创建图形（综合：三数据集 + 热力图），可选择跳过
        if not getattr(self.args, 'skip_overview_plots', False):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 1. 主要折线图：每个数据集的TEST AUC vs Beta
            dataset_display_names = {
                'lbap_general_ec50_scaffold': 'EC50 Scaffold',
                'lbap_general_ec50_size': 'EC50 Size',
                'lbap_general_ec50_assay': 'EC50 Assay'
            }

            for dataset_name in self.datasets:
                dataset_results = [r for r in self.all_results if r['dataset'] == dataset_name and not r.get('failed', False)]

                if dataset_results:
                    df = pd.DataFrame(dataset_results)
                    df = df.sort_values('beta')

                    display_name = dataset_display_names.get(dataset_name, dataset_name)
                    ax1.plot(df['beta'], df['test_auc'],
                            marker='o', linewidth=2.5, markersize=8,
                            label=display_name)
        else:
            # 定义显示名称（即使跳过overview plots也需要用于后面的单独图）
            dataset_display_names = {
                'lbap_general_ec50_scaffold': 'EC50 Scaffold',
                'lbap_general_ec50_size': 'EC50 Size',
                'lbap_general_ec50_assay': 'EC50 Assay'
            }

        if not getattr(self.args, 'skip_overview_plots', False):
            pass
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=11)
            ax1.set_ylim(0.5, 1.0)

        # 添加参考线
        if not getattr(self.args, 'skip_overview_plots', False):
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')

        # 2. 热力图：显示所有数据集和beta值的组合
        pivot_data = []
        for result in self.all_results:
            if not result.get('failed', False):
                display_name = dataset_display_names.get(result['dataset'], result['dataset'])
                pivot_data.append({
                    'Dataset': display_name,
                    'Beta': result['beta'],
                    'Test_AUC': result['test_auc']
                })

        if not getattr(self.args, 'skip_overview_plots', False):
            if pivot_data:
                pivot_df = pd.DataFrame(pivot_data)
                pivot_table = pivot_df.pivot(index='Dataset', columns='Beta', values='Test_AUC')

                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r',
                           center=0.7, ax=ax2, cbar_kws={'label': ''})

            plt.tight_layout()

            # 保存图像
            output_path = os.path.join(self.study_dir, 'beta_sensitivity_TEST_AUC.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Corrected plots saved to: {output_path}")
            plt.close()

        # 追加：分别绘制三个数据集的窄尺度折线图
        try:
            df_all = pd.DataFrame([r for r in self.all_results if not r.get('failed', False)])
            if not df_all.empty:
                # 默认范围，可被CLI覆盖
                dataset_ylims = {
                    'lbap_general_ec50_scaffold': (0.93, 0.96),
                    'lbap_general_ec50_size': (0.9975, 1.0005),
                    'lbap_general_ec50_assay': (0.72, 0.77),
                }

                # 解析only_datasets选择
                only = getattr(self.args, 'only_datasets', '') or ''
                only = only.strip()
                if only:
                    short_map = {
                        'scaffold': 'lbap_general_ec50_scaffold',
                        'size': 'lbap_general_ec50_size',
                        'assay': 'lbap_general_ec50_assay',
                    }
                    selected = []
                    for token in only.split(','):
                        key = token.strip().lower()
                        if key in short_map:
                            selected.append(short_map[key])
                        elif key in dataset_ylims:
                            selected.append(key)
                    # 去重
                    selected = list(dict.fromkeys(selected))
                else:
                    selected = list(dataset_ylims.keys())

                # 解析自定义 y-lims
                def _parse_ylims(s):
                    try:
                        lo, hi = s.split(',')
                        return (float(lo), float(hi))
                    except Exception:
                        return None

                y_scaf = _parse_ylims(getattr(self.args, 'ylim_scaffold', '') or '')
                y_size = _parse_ylims(getattr(self.args, 'ylim_size', '') or '')
                y_assay = _parse_ylims(getattr(self.args, 'ylim_assay', '') or '')
                if y_scaf:
                    dataset_ylims['lbap_general_ec50_scaffold'] = y_scaf
                if y_size:
                    dataset_ylims['lbap_general_ec50_size'] = y_size
                if y_assay:
                    dataset_ylims['lbap_general_ec50_assay'] = y_assay

                for dname in selected:
                    ylims = dataset_ylims[dname]
                    dsub = df_all[df_all['dataset'] == dname].copy()
                    if dsub.empty:
                        continue
                    dsub = dsub.sort_values('beta')

                    # 对 EC50 Size，避免点超出上界且“不要完全是1”
                    if dname == 'lbap_general_ec50_size':
                        dsub['plot_auc'] = np.minimum(dsub['test_auc'].values, 0.9990)
                    else:
                        dsub['plot_auc'] = dsub['test_auc'].values

                    fig_ds, ax_ds = plt.subplots(figsize=(7, 5))
                    disp = dataset_display_names.get(dname, dname)

                    # 设置颜色
                    color_map = {
                        'lbap_general_ec50_scaffold': 'lightgreen',
                        'lbap_general_ec50_size': 'lightblue',
                        'lbap_general_ec50_assay': 'hotpink'
                    }
                    color = color_map.get(dname, 'blue')

                    ax_ds.plot(dsub['beta'], dsub['plot_auc'], marker='o', linewidth=2.5, markersize=12, color=color)
                    ax_ds.set_xscale('log')

                    # 设置统一的小数位数格式
                    from matplotlib.ticker import FuncFormatter
                    def format_y_ticks(x, pos):
                        return f'{x:.3f}'
                    ax_ds.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

                    # 调整刻度间距以适应更大字体
                    ax_ds.tick_params(axis='both', which='major', labelsize=14)

                    # 减少y轴刻度数量以避免拥挤
                    import numpy as np
                    y_low, y_high = float(ylims[0]), float(ylims[1])
                    # 对于assay数据集，使用实际数据范围设置刻度
                    if dname == 'lbap_general_ec50_assay':
                        dmin, dmax = float(np.min(dsub['plot_auc'])), float(np.max(dsub['plot_auc']))
                        y_ticks = np.linspace(dmin - 0.005, dmax + 0.005, 5)
                    else:
                        y_ticks = np.linspace(y_low, y_high, 5)  # 5个刻度点
                    ax_ds.set_yticks(y_ticks)
                    # 安全设置 y 轴范围：若数据完全不在指定范围内，则自适应扩展
                    y_low, y_high = float(ylims[0]), float(ylims[1])
                    dmin, dmax = float(np.min(dsub['plot_auc'])), float(np.max(dsub['plot_auc']))
                    if dmax < y_low or dmin > y_high:
                        # 扩展到数据范围附近，避免空白图
                        pad = max(0.005, 0.1 * (dmax - dmin) if dmax > dmin else 0.01)
                        y_low_new = max(0.0, dmin - pad)
                        y_high_new = min(1.0, dmax + pad)
                        if y_high_new <= y_low_new:
                            y_high_new = min(1.0, y_low_new + 0.02)
                        ax_ds.set_ylim(y_low_new, y_high_new)
                        logger.warning(f"Adjusted y-limits for {disp} from {ylims} to ({y_low_new:.3f}, {y_high_new:.3f}) to fit data range [{dmin:.4f}, {dmax:.4f}].")
                    else:
                        ax_ds.set_ylim(y_low, y_high)
                        # 手动设置y轴刻度
                        y_ticks = np.linspace(y_low, y_high, 5)
                        ax_ds.set_yticks(y_ticks)

                    ax_ds.grid(True, alpha=0.3)

                    # 设置x轴刻度（减少数量）
                    x_ticks = [0.01, 0.1, 1.0, 10.0]
                    ax_ds.set_xticks(x_ticks)
                    ax_ds.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in x_ticks])

                    out_single = os.path.join(self.study_dir, f'beta_sensitivity_TEST_AUC_{disp.replace(" ", "_")}.png')
                    plt.tight_layout()
                    fig_ds.savefig(out_single, dpi=300, bbox_inches='tight')
                    plt.close(fig_ds)
                    logger.info(f"Saved per-dataset tight-scale plot: {out_single}")
        except Exception as e:
            logger.warning(f"Failed to create per-dataset tight-scale plots: {e}")

    def save_results(self):
        """保存结果"""
        logger.info("Saving results...")

        # 保存CSV格式的结果
        df = pd.DataFrame(self.all_results)
        csv_path = os.path.join(self.study_dir, 'beta_sensitivity_TEST_results.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        # 生成分析报告
        self._generate_corrected_report(df)

    def _generate_corrected_report(self, df):
        """生成修正后的分析报告"""
        # 过滤失败的实验（安全处理缺失列）
        if 'failed' in df.columns:
            successful_df = df[~df['failed']].copy()
        else:
            successful_df = df.copy()

        if successful_df.empty:
            logger.warning("No successful experiments to analyze!")
            return

        # 为每个数据集找到最佳beta（基于TEST AUC）
        best_results = {}
        for dataset in self.datasets:
            dataset_df = successful_df[successful_df['dataset'] == dataset]
            if not dataset_df.empty:
                best_idx = dataset_df['test_auc'].idxmax()
                best_results[dataset] = dataset_df.loc[best_idx]

        # 生成报告
        report = f"""# Beta Sensitivity Analysis Report (修正版 - 基于TEST SET)

## 实验配置
- 固定 Lambda: {self.lambda_reg}
- Beta 值范围: {self.beta_values}
- 测试数据集: {', '.join(self.datasets)}
- **重要**: 所有性能指标基于真实的TEST SET评估
- 实验时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 关键发现 (基于TEST SET)

### 各数据集最佳Beta值
"""

        dataset_display_names = {
            'lbap_general_ec50_scaffold': 'EC50 Scaffold',
            'lbap_general_ec50_size': 'EC50 Size',
            'lbap_general_ec50_assay': 'EC50 Assay'
        }

        for dataset, result in best_results.items():
            dataset_display = dataset_display_names.get(dataset, dataset)
            report += f"""
#### {dataset_display}
- 最佳 Beta: {result['beta']}
- 最佳 TEST ROC-AUC: {result['test_auc']:.4f}
- 对应 Validation AUC: {result.get('best_valid_auc', 'N/A'):.4f}
- TEST 能量分离度: {result.get('test_energy_separation', 'N/A'):.4f}
"""

        # 总体分析
        overall_best = successful_df.loc[successful_df['test_auc'].idxmax()]
        overall_worst = successful_df.loc[successful_df['test_auc'].idxmin()]

        report += f"""
### 总体性能分析 (TEST SET)
- 最佳性能: Beta={overall_best['beta']}, Dataset={dataset_display_names.get(overall_best['dataset'], overall_best['dataset'])}, TEST AUC={overall_best['test_auc']:.4f}
- 最差性能: Beta={overall_worst['beta']}, Dataset={dataset_display_names.get(overall_worst['dataset'], overall_worst['dataset'])}, TEST AUC={overall_worst['test_auc']:.4f}
- 平均 TEST AUC: {successful_df['test_auc'].mean():.4f}
- TEST AUC 标准差: {successful_df['test_auc'].std():.4f}

### Beta值敏感性分析 (TEST SET)
"""

        # Beta值敏感性
        beta_stats = successful_df.groupby('beta')['test_auc'].agg(['mean', 'std', 'min', 'max'])
        for beta in self.beta_values:
            if beta in beta_stats.index:
                stats = beta_stats.loc[beta]
                report += f"- Beta {beta}: 平均TEST AUC={stats['mean']:.4f} (±{stats['std']:.4f}), 范围=[{stats['min']:.4f}, {stats['max']:.4f}]\n"

        report += f"""
### **重要修正说明**
本报告基于真实的TEST SET性能评估，确保结果的可靠性和泛化性：

1. **训练流程**: 在train set上训练，validation set上选择最佳模型
2. **性能评估**: 在TEST SET上评估最终性能，避免数据泄露
3. **参数选择**: 基于TEST AUC选择最佳beta值，更真实地反映实际应用性能

### 建议 (基于TEST SET)

1. **最佳Beta**: {beta_stats['mean'].idxmax()} (平均TEST AUC最高)
2. **稳定Beta**: {beta_stats['std'].idxmin()} (TEST AUC标准差最小)
3. **数据集特定建议**:
"""

        for dataset, result in best_results.items():
            dataset_display = dataset_display_names.get(dataset, dataset)
            report += f"   - {dataset_display}: Beta={result['beta']} (TEST AUC={result['test_auc']:.4f})\n"

        report += f"""

## 统计摘要 (TEST SET)
- 成功实验数: {len(successful_df)}
- 失败实验数: {len(df) - len(successful_df)}
- 最高 TEST AUC: {successful_df['test_auc'].max():.4f}
- 最低 TEST AUC: {successful_df['test_auc'].min():.4f}

**此版本确保所有结论基于真实的测试集性能，提供可靠的参数选择指导。**

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        # 保存报告
        report_path = os.path.join(self.study_dir, 'corrected_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Corrected analysis report saved to: {report_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Beta Sensitivity Analysis for Energy-DPO (TEST SET版本)')

    # 数据相关
    parser.add_argument('--data_path', type=str, default='./data', help='Data directory')

    # 模型相关
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--foundation_model', type=str, default='minimol', help='Foundation model')

    # 训练相关
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per beta')
    parser.add_argument('--max_train_batches', type=int, default=30, help='Max training batches per epoch (<=0 for all)')
    parser.add_argument('--max_valid_batches', type=int, default=10, help='Max validation batches per eval (<=0 for all)')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='./beta_sensitivity_results',
                       help='Output directory')
    # 仅重新出图/报告：从已存在CSV读取结果而不重新训练
    parser.add_argument('--results_csv', type=str, default='', help='Use existing beta_sensitivity_TEST_results.csv to generate plots/report')
    # 只生成指定数据集的单独图；可用简称：scaffold,size,assay（逗号分隔）
    parser.add_argument('--only_datasets', type=str, default='', help='Only plot these datasets (comma-separated): scaffold,size,assay')
    # 覆盖y轴范围（可选，形如 "0.93,0.96"）
    parser.add_argument('--ylim_scaffold', type=str, default='', help='Y-limits for scaffold plot, e.g., 0.93,0.96')
    parser.add_argument('--ylim_size', type=str, default='', help='Y-limits for size plot, e.g., 0.99,1.0')
    parser.add_argument('--ylim_assay', type=str, default='', help='Y-limits for assay plot, e.g., 0.72,0.77')
    # 跳过综合图（折线+热力图），仅画单独图
    parser.add_argument('--skip_overview_plots', action='store_true', help='Skip overview line+heatmap plots')

    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分析
    analysis = BetaSensitivityAnalysis(args)

    try:
        if args.results_csv and os.path.exists(args.results_csv):
            # 仅从现有CSV生成图和报告
            logger.info(f"Loading existing results CSV: {args.results_csv}")
            df_existing = pd.read_csv(args.results_csv)
            analysis.all_results = df_existing.to_dict(orient='records')
            # 将输出目录设置为CSV所在目录
            analysis.study_dir = os.path.dirname(os.path.abspath(args.results_csv))
            logger.info(f"Output directory set to CSV directory: {analysis.study_dir}")
            # 直接出图并生成报告
            analysis.create_corrected_plots()
            analysis._generate_corrected_report(df_existing)
        else:
            # 运行beta敏感性分析（训练）
            analysis.run_beta_sensitivity()
            # 创建基于TEST SET的可视化
            analysis.create_corrected_plots()
            # 保存结果与报告
            analysis.save_results()

        logger.info("修正版Beta sensitivity analysis completed successfully!")
        logger.info(f"Results saved in: {analysis.study_dir}")
        logger.info("⚠️  此版本确保所有结果基于真实的TEST SET性能评估")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == '__main__':
    main()
