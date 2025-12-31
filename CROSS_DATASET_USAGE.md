# 跨数据集评估使用指南

## 功能简介

现在支持在一个数据集上训练/验证，在另一个数据集上测试！

**典型场景：** 在 EC50 Scaffold（基于化学骨架的分布偏移）上训练，在 EC50 Size（基于分子大小的分布偏移）上测试，验证模型的跨域泛化能力。

## 修改的文件

### 1. `main.py`
添加了两个新参数：
- `--test_data_file`: 独立测试数据文件路径
- `--test_drugood_subset`: 测试数据集名称

### 2. `data_loader.py`
- 新增 `_load_test_data()` 方法：从独立文件加载测试数据
- 修改 `_check_cross_split_overlap()`：跨数据集模式下跳过训练-测试重叠检查
- 修改 `__init__` 和 `_load_raw_data()`：支持新参数

### 3. `evaluation.py`
在 `parse_args()` 中添加了相同的两个参数。

## 使用方法

### 步骤 1: 训练模型（在 EC50 Scaffold 上）

```bash
python main.py \
  --mode train \
  --dataset lbap_general_ec50_scaffold \
  --data_file ./data/raw/lbap_general_ec50_scaffold.json \
  --foundation_model minimol \
  --output_dir ./outputs/minimol/ec50_scaffold/1 \
  --seed 1 \
  --data_seed 42 \
  --epochs 500 \
  --batch_size 512 \
  --lr 1e-4
```

### 步骤 2: 评估模型（在 EC50 Size 上测试）

```bash
python main.py \
  --mode eval \
  --dataset lbap_general_ec50_scaffold \
  --data_file ./data/raw/lbap_general_ec50_scaffold.json \
  --test_data_file ./data/raw/lbap_general_ec50_size.json \
  --test_drugood_subset lbap_general_ec50_size \
  --foundation_model minimol \
  --model_path ./outputs/minimol/ec50_scaffold/1/best_model.pth \
  --output_dir ./outputs/minimol/ec50_scaffold/1 \
  --seed 1 \
  --data_seed 42
```

**关键参数说明：**
- `--dataset`: 训练数据集（必须与训练时一致）
- `--data_file`: 训练数据文件（用于加载训练时的验证集）
- `--test_data_file`: **新！** 测试数据文件路径
- `--test_drugood_subset`: **新！** 测试数据集名称

### 步骤 3: 使用 evaluation.py 脚本

也可以使用独立的评估脚本：

```bash
python evaluation.py \
  --model_path ./outputs/minimol/ec50_scaffold/1/best_model.pth \
  --dataset lbap_general_ec50_scaffold \
  --drugood_subset lbap_general_ec50_scaffold \
  --test_data_file ./data/raw/lbap_general_ec50_size.json \
  --test_drugood_subset lbap_general_ec50_size \
  --output_dir ./evaluation_results/scaffold_to_size
```

## 数据集配置建议

### 训练阶段（EC50 Scaffold）
```python
# 在 data_loader.py 第 249-253 行修改
default_sizes = {
    'train_id': 2000,      # 保持不变
    'train_ood': 2000,     # 或测试不同比例: 200 (10:1), 100 (20:1), 40 (50:1)
    'val_id': 600,         # 保持不变
    'val_ood': 600,        # 保持不变
    'test_id': 1000,       # 不影响（会被 Size 数据覆盖）
    'test_ood': 1000       # 不影响（会被 Size 数据覆盖）
}
```

### 测试阶段（EC50 Size）
- 测试集会从 `lbap_general_ec50_size.json` 加载
- 推荐使用完整测试集（~14,257 ID + ~20,312 OOD）以获得可靠评估
- 代码会自动采样到 `default_sizes` 中设置的大小

## 重要特性

### ✅ 数据泄漏防护
- **同数据集模式**：严格检查训练-测试重叠
- **跨数据集模式**：自动跳过训练-测试重叠检查（因为来自不同数据集）
- 仍然检查 ID-OOD 重叠、训练-验证重叠

### ✅ 缓存管理
- 训练数据缓存：`lbap_general_ec50_scaffold_seed42_splits.json`
- 特征缓存：独立管理 Scaffold 和 Size 的特征缓存
- 测试数据：每次从原始文件加载（不采样，使用完整测试集）

### ✅ 向后兼容
- 不指定 `--test_data_file` 时，行为与之前完全相同
- 测试数据来自训练数据集的 test_id 和 test_ood

## 测试验证

运行测试脚本验证功能：

```bash
python test_cross_dataset.py
```

应该看到：
```
✓ ALL TESTS PASSED!
Cross-dataset evaluation is working correctly.
```

## 实验建议

### 实验 1: ID:OOD 比例对跨域泛化的影响

在 Scaffold 上训练（不同 ID:OOD 比例），在 Size 上测试：

```bash
# 比例 1:1 (baseline)
修改 data_loader.py: train_ood = 2000
python main.py --mode train ...
python main.py --mode eval --test_data_file ./data/raw/lbap_general_ec50_size.json ...

# 比例 10:1
修改 data_loader.py: train_ood = 200
python main.py --mode train ...
python main.py --mode eval --test_data_file ./data/raw/lbap_general_ec50_size.json ...

# 比例 20:1
修改 data_loader.py: train_ood = 100
python main.py --mode train ...
python main.py --mode eval --test_data_file ./data/raw/lbap_general_ec50_size.json ...

# 比例 50:1
修改 data_loader.py: train_ood = 40
python main.py --mode train ...
python main.py --mode eval --test_data_file ./data/raw/lbap_general_ec50_size.json ...
```

### 实验 2: 对比域内和跨域性能

```bash
# 域内测试（Scaffold → Scaffold）
python main.py --mode eval --dataset lbap_general_ec50_scaffold ...

# 跨域测试（Scaffold → Size）
python main.py --mode eval --dataset lbap_general_ec50_scaffold \
  --test_data_file ./data/raw/lbap_general_ec50_size.json \
  --test_drugood_subset lbap_general_ec50_size ...

# 跨域测试（Scaffold → Assay）
python main.py --mode eval --dataset lbap_general_ec50_scaffold \
  --test_data_file ./data/raw/lbap_general_ec50_assay.json \
  --test_drugood_subset lbap_general_ec50_assay ...
```

## 注意事项

1. **数据集类型必须兼容**：目前支持 DrugOOD (lbap_*) 和 GOOD 数据集之间的跨数据集测试
2. **Foundation model 必须一致**：训练和测试使用相同的分子编码器（minimol/unimol）
3. **验证集始终来自训练数据集**：这样可以在训练域上进行超参数调优
4. **测试集完全来自目标数据集**：确保真正测试跨域泛化能力

## 故障排查

### 问题：找不到测试数据文件
```
FileNotFoundError: Could not find test data file: ...
```
**解决**：检查 `--test_data_file` 路径是否正确

### 问题：数据泄漏错误（即使在跨数据集模式）
```
ValueError: Cross-split overlap detected!
```
**可能原因**：
- ID 和 OOD 测试集之间有重叠（这是真实的数据问题）
- 训练和验证集之间有重叠（这是真实的数据问题）

### 问题：测试集大小不符合预期
**检查**：`data_loader.py` 中的 `default_sizes` 设置，测试集会被采样到这个大小

## 总结

现在你可以：
✅ 在 EC50 Scaffold 上训练
✅ 在 EC50 Size 上测试
✅ 测试不同 ID:OOD 训练比例的跨域泛化能力
✅ 自动处理数据加载和缓存
✅ 保持完整的数据泄漏保护

Good luck with your experiments! 🚀
