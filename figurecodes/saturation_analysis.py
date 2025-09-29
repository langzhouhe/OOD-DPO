#!/usr/bin/env python3
"""
饱和与梯度弱化现象分析工具
专门用于分析和可视化 Energy-DPO 中的饱和和梯度弱化现象
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import argparse
import os

class SaturationAnalysis:
    """分析DPO损失函数的饱和和梯度弱化现象"""

    def __init__(self, output_dir="./saturation_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_dpo_function(self):
        """分析DPO损失函数的数学性质"""
        print("Analyzing DPO loss function mathematical properties...")

        # 定义能量差范围
        energy_diff = np.linspace(-10, 10, 1000)
        beta_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. DPO损失函数值
        ax1 = axes[0, 0]
        for beta in beta_values:
            dpo_loss = F.softplus(-beta * torch.tensor(energy_diff)).numpy()
            ax1.plot(energy_diff, dpo_loss, label=f'β={beta}', linewidth=2)

        ax1.set_xlabel('Energy Difference (OOD - ID)')
        ax1.set_ylabel('DPO Loss Value')
        ax1.set_title('DPO Loss Function: softplus(-β × diff)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect separation')

        # 添加饱和区域标注
        ax1.add_patch(Rectangle((-10, 0), 5, 1, alpha=0.2, color='red', label='Saturation'))
        ax1.add_patch(Rectangle((5, 0), 5, 1, alpha=0.2, color='blue', label='Good separation'))

        # 2. DPO损失梯度（对能量差的梯度）
        ax2 = axes[0, 1]
        for beta in beta_values:
            # 梯度 = -β * sigmoid(-β * energy_diff)
            gradient = -beta * torch.sigmoid(-beta * torch.tensor(energy_diff)).numpy()
            ax2.plot(energy_diff, np.abs(gradient), label=f'β={beta}', linewidth=2)

        ax2.set_xlabel('Energy Difference (OOD - ID)')
        ax2.set_ylabel('|Gradient| w.r.t Energy Diff')
        ax2.set_title('DPO Loss Gradient Magnitude')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # 3. 不同beta下的sigmoid函数
        ax3 = axes[1, 0]
        for beta in beta_values:
            sigmoid_vals = torch.sigmoid(beta * torch.tensor(energy_diff)).numpy()
            ax3.plot(energy_diff, sigmoid_vals, label=f'β={beta}', linewidth=2)

        ax3.set_xlabel('Energy Difference (OOD - ID)')
        ax3.set_ylabel('σ(β × diff)')
        ax3.set_title('Sigmoid Function with Different β')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # 4. 梯度衰减分析
        ax4 = axes[1, 1]
        # 分析在不同能量分离度下的梯度衰减
        separations = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

        for sep in separations:
            gradients = []
            for beta in beta_values:
                grad = beta * torch.sigmoid(-beta * torch.tensor(sep)).item()
                gradients.append(grad)
            ax4.plot(beta_values, gradients, 'o-', label=f'Separation={sep}')

        ax4.set_xlabel('Beta (β)')
        ax4.set_ylabel('Gradient Magnitude')
        ax4.set_title('Gradient vs Beta for Different Energy Separations')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dpo_mathematical_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_gradient_flow(self):
        """分析梯度流动的问题"""
        print("Analyzing gradient flow patterns...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 模拟训练过程中的梯度变化
        epochs = np.arange(0, 50)

        # 不同beta值下的梯度演化模式
        beta_scenarios = [
            (0.01, "Very Small β - Weak Gradients"),
            (0.1, "Optimal β - Balanced"),
            (1.0, "Large β - Risk of Saturation"),
            (5.0, "Very Large β - Saturation"),
        ]

        lambda_values = [0.01, 0.1]

        for idx, (beta, title) in enumerate(beta_scenarios):
            ax = axes[idx // 2, idx % 2] if idx < 4 else None
            if ax is None:
                continue

            for lambda_reg in lambda_values:
                # 模拟梯度演化（基于理论分析）
                if beta < 0.05:  # 梯度过小
                    grad_norm = 0.1 * np.exp(-0.05 * epochs) + 0.01 * np.random.normal(0, 0.02, len(epochs))
                elif beta > 2.0:  # 饱和
                    grad_norm = 1.0 * np.exp(-0.3 * epochs) + 0.001 * np.random.normal(0, 0.1, len(epochs))
                    grad_norm = np.maximum(grad_norm, 1e-6)  # 饱和到极小值
                else:  # 正常情况
                    grad_norm = (0.5 + 0.3 * np.sin(0.2 * epochs)) * np.exp(-0.1 * epochs) + \
                               0.05 * np.random.normal(0, 0.1, len(epochs))

                # 添加L2正则化的影响
                if lambda_reg > 0.05:
                    grad_norm *= 0.7  # 强正则化降低梯度

                ax.plot(epochs, np.maximum(grad_norm, 1e-8),
                       label=f'λ={lambda_reg}', linewidth=2)

            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(title)
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 标注问题区域
            if beta < 0.05:
                ax.axhline(y=1e-5, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
            elif beta > 2.0:
                ax.axhspan(1e-6, 1e-4, alpha=0.2, color='red', label='Saturation zone')

        # 额外的分析图：梯度-损失关系
        ax_extra1 = axes[1, 2]

        # 理想 vs 问题情况的梯度-损失曲线
        loss_values = np.linspace(0.1, 5.0, 100)

        # 正常情况：梯度与损失成反比
        normal_grads = 1.0 / (loss_values + 0.1)

        # 梯度消失：梯度过小
        vanishing_grads = 0.01 * normal_grads

        # 梯度爆炸：梯度不稳定
        exploding_grads = normal_grads * (1 + 2 * np.sin(10 * loss_values))

        ax_extra1.plot(loss_values, normal_grads, 'g-', linewidth=3, label='Healthy Training')
        ax_extra1.plot(loss_values, vanishing_grads, 'b--', linewidth=3, label='Gradient Vanishing')
        ax_extra1.plot(loss_values, np.abs(exploding_grads), 'r:', linewidth=3, label='Gradient Instability')

        ax_extra1.set_xlabel('Training Loss')
        ax_extra1.set_ylabel('Gradient Norm')
        ax_extra1.set_title('Gradient-Loss Relationship Patterns')
        ax_extra1.set_yscale('log')
        ax_extra1.legend()
        ax_extra1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_flow_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_parameter_guidance_chart(self):
        """创建参数选择指导图表"""
        print("Creating parameter selection guidance chart...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Beta 选择指导
        ax1 = axes[0, 0]
        beta_range = np.logspace(-2, 1, 100)

        # 不同能量分离度下的表现
        separations = [0.5, 1.0, 2.0, 5.0]
        colors = ['red', 'orange', 'green', 'blue']

        for sep, color in zip(separations, colors):
            # 模拟性能得分（基于理论分析）
            performance = []
            for beta in beta_range:
                if beta < 0.05:
                    score = 0.3 + 0.1 * np.log(beta + 0.01)  # 梯度太弱
                elif beta > 2.0:
                    score = 0.8 * np.exp(-(beta - 1.0)**2 / 2.0)  # 饱和惩罚
                else:
                    score = 0.9 - 0.1 * (beta - 0.5)**2  # 最优区域

                # 考虑能量分离度的影响
                if sep > 2.0:
                    score *= 1.1  # 大分离度更容易训练
                elif sep < 1.0:
                    score *= 0.8  # 小分离度更难训练

                performance.append(max(0.1, min(1.0, score)))

            ax1.plot(beta_range, performance, color=color, linewidth=2,
                    label=f'Energy Sep = {sep}')

        ax1.set_xlabel('Beta (β)')
        ax1.set_ylabel('Expected Performance Score')
        ax1.set_title('Beta Selection Guide')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvspan(0.05, 0.5, alpha=0.2, color='green', label='Recommended range')

        # 2. Lambda 选择指导
        ax2 = axes[0, 1]
        lambda_range = np.logspace(-3, 0, 100)

        # 不同训练难度下的稳定性
        difficulties = ['Easy', 'Medium', 'Hard']
        colors = ['green', 'orange', 'red']

        for diff, color in zip(difficulties, colors):
            stability = []
            for lam in lambda_range:
                if lam < 0.005:
                    score = 0.5 + 0.2 * np.log(lam + 0.001)  # 正则化太弱
                elif lam > 0.2:
                    score = 0.9 * np.exp(-10 * (lam - 0.1))  # 正则化太强
                else:
                    score = 0.95 - 0.1 * (lam - 0.05)**2  # 最优区域

                # 考虑训练难度
                if diff == 'Hard':
                    score *= 0.8  # 困难情况需要更强正则化
                elif diff == 'Easy':
                    score *= 1.1  # 简单情况允许弱正则化

                stability.append(max(0.1, min(1.0, score)))

            ax2.plot(lambda_range, stability, color=color, linewidth=2, label=diff)

        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Training Stability Score')
        ax2.set_title('Lambda Selection Guide')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvspan(0.01, 0.1, alpha=0.2, color='green', label='Recommended range')

        # 3. 联合参数空间分析
        ax3 = axes[1, 0]

        beta_grid = np.logspace(-2, 1, 50)
        lambda_grid = np.logspace(-3, 0, 50)
        B, L = np.meshgrid(beta_grid, lambda_grid)

        # 模拟整体性能得分
        overall_score = np.zeros_like(B)
        for i in range(len(lambda_grid)):
            for j in range(len(beta_grid)):
                beta, lam = B[i, j], L[i, j]

                # Beta 贡献
                if beta < 0.05:
                    beta_score = 0.3
                elif beta > 2.0:
                    beta_score = 0.5
                else:
                    beta_score = 0.9

                # Lambda 贡献
                if lam < 0.005 or lam > 0.2:
                    lambda_score = 0.4
                else:
                    lambda_score = 0.9

                # 组合得分
                overall_score[i, j] = (beta_score + lambda_score) / 2

        contour = ax3.contourf(B, L, overall_score, levels=20, cmap='RdYlGn')
        ax3.set_xlabel('Beta (β)')
        ax3.set_ylabel('Lambda (λ)')
        ax3.set_title('Joint Parameter Performance Map')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        plt.colorbar(contour, ax=ax3, label='Performance Score')

        # 标注推荐区域
        ax3.axvspan(0.05, 0.5, alpha=0.3, color='white', label='Recommended β')
        ax3.axhspan(0.01, 0.1, alpha=0.3, color='white', label='Recommended λ')

        # 4. 问题诊断流程图
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.9, 'Parameter Problem Diagnosis',
                ha='center', va='center', fontsize=16, fontweight='bold',
                transform=ax4.transAxes)

        diagnostic_text = """
        Symptoms → Likely Cause → Solution

        • Loss decreases very slowly
          → β too small (< 0.05)
          → Increase β to 0.1-0.5

        • Training becomes unstable
          → λ too small (< 0.01)
          → Increase λ to 0.01-0.1

        • Loss plateaus early
          → β too large (> 2.0)
          → Decrease β to 0.1-1.0

        • Overfitting quickly
          → λ too small
          → Increase λ or add dropout

        • Underfitting
          → λ too large (> 0.2)
          → Decrease λ to 0.01-0.05
        """

        ax4.text(0.05, 0.8, diagnostic_text, ha='left', va='top',
                fontsize=10, transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_guidance_chart.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("Generating comprehensive analysis report...")

        report = """
# Energy-DPO 饱和与梯度分析报告

## 1. DPO 损失函数特性

### 数学形式
```
L_DPO = softplus(-β × (E_ood - E_id))
其中：softplus(x) = log(1 + exp(x))
```

### 关键特性
- **单调性**: 当 E_ood > E_id 时损失减小
- **饱和性**: 极大的能量差会导致梯度趋近于零
- **敏感性**: β 控制对能量差的敏感程度

## 2. Beta (β) 参数影响分析

### β 过小 (< 0.05)
**现象**: 梯度弱化
- 损失函数对能量差不敏感
- 梯度信号微弱，训练缓慢
- 能量分离度提升困难

**诊断指标**:
- 梯度范数 < 1e-5
- 损失下降极慢
- 能量分离度长期无改善

### β 过大 (> 2.0)
**现象**: 饱和效应
- sigmoid 函数快速饱和
- 梯度消失，训练停滞
- 对小的能量差过度敏感

**诊断指标**:
- 梯度范数快速衰减到 1e-6
- 损失提前平台化
- 训练曲线呈现早期饱和

### β 最优范围: 0.1 - 1.0
- 提供适度的梯度信号
- 平衡敏感性和稳定性
- 允许渐进式能量分离

## 3. Lambda (λ) 正则化分析

### L2 正则化作用机制
```
L_total = L_DPO + λ × (||E_id||² + ||E_ood||²)
```

### λ 过小 (< 0.01)
**问题**: 能量爆炸
- 能量值无约束增长
- 训练不稳定
- 可能出现梯度爆炸

### λ 过大 (> 0.2)
**问题**: 过度约束
- 能量被压制到接近零
- 分离能力受限
- 可能导致欠拟合

### λ 最优范围: 0.01 - 0.1
- 提供适度的能量约束
- 维持训练稳定性
- 保持足够的分离能力

## 4. 联合优化策略

### 阶段性调整策略
1. **初期** (Epoch 1-10): β=0.1, λ=0.05
   - 建立基础能量分离
   - 确保训练稳定性

2. **中期** (Epoch 11-30): β=0.2-0.5, λ=0.02
   - 增强分离信号
   - 减少正则化约束

3. **后期** (Epoch 31+): β=0.1-0.3, λ=0.01
   - 精细调优
   - 避免过拟合

### 自适应调整规则
```python
if gradient_norm < 1e-5:
    β *= 2.0  # 增强信号
elif gradient_norm > 10.0:
    λ *= 2.0  # 增强正则化
```

## 5. 实践建议

### 快速诊断清单
- [ ] 梯度范数在 [1e-5, 10] 范围内？
- [ ] 损失稳定下降？
- [ ] 能量分离度持续改善？
- [ ] 训练过程无异常震荡？

### 参数调优序列
1. 先固定 λ=0.05，调优 β
2. 找到最佳 β 后，调优 λ
3. 联合微调两个参数
4. 监控训练稳定性

### 常见问题解决方案
| 症状 | 原因 | 解决方案 |
|------|------|----------|
| 训练过慢 | β 太小 | 增加 β 到 0.1-0.5 |
| 早期饱和 | β 太大 | 减少 β 到 0.1-1.0 |
| 训练不稳定 | λ 太小 | 增加 λ 到 0.01-0.1 |
| 欠拟合 | λ 太大 | 减少 λ 到 0.005-0.02 |

生成时间: {datetime_str}
"""

        with open(os.path.join(self.output_dir, 'saturation_analysis_report.md'), 'w') as f:
            from datetime import datetime
            f.write(report.format(datetime_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def run_complete_analysis(self):
        """运行完整的饱和分析"""
        print("Starting comprehensive saturation and gradient analysis...")

        self.analyze_dpo_function()
        self.analyze_gradient_flow()
        self.create_parameter_guidance_chart()
        self.generate_comprehensive_report()

        print(f"Analysis complete! Results saved in {self.output_dir}")
        print(f"Key files generated:")
        print(f"  - dpo_mathematical_analysis.png")
        print(f"  - gradient_flow_analysis.png")
        print(f"  - parameter_guidance_chart.png")
        print(f"  - saturation_analysis_report.md")

def main():
    parser = argparse.ArgumentParser(description='Saturation and Gradient Analysis for Energy-DPO')
    parser.add_argument('--output_dir', type=str, default='./saturation_analysis',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    analyzer = SaturationAnalysis(args.output_dir)
    analyzer.run_complete_analysis()

if __name__ == '__main__':
    main()