# 如何在原始版本和跨数据集版本之间切换

## 📁 文件说明

### 原始版本（当前使用）
- `data_loader.py` - 原始数据加载器
- `main.py` - 原始主程序
- `evaluation.py` - 原始评估脚本
- `run_experiments.sh` - 原始实验脚本

### 跨数据集泛化版本（备份）
- `data_loader_generalization.py` - 支持跨数据集测试
- `main_generalization.py` - 添加了 --test_data_file 参数
- `evaluation_generalization.py` - 添加了 --test_data_file 参数
- `run_cross_dataset_experiments.sh` - 跨数据集实验脚本
- `CROSS_DATASET_USAGE.md` - 详细使用文档
- `test_cross_dataset.py` - 测试脚本

## 🔄 切换到跨数据集版本

### 方法 1: 手动复制（推荐）

```bash
# 切换到跨数据集版本
cp data_loader_generalization.py data_loader.py
cp main_generalization.py main.py
cp evaluation_generalization.py evaluation.py

# 运行跨数据集实验
./run_cross_dataset_experiments.sh
```

### 方法 2: 使用 Git

```bash
# 暂存当前修改
git stash

# 复制generalization文件
cp *_generalization.py tmp/
cp tmp/data_loader_generalization.py data_loader.py
cp tmp/main_generalization.py main.py
cp tmp/evaluation_generalization.py evaluation.py

# 运行实验
./run_cross_dataset_experiments.sh
```

## 🔙 恢复到原始版本

### 方法 1: 使用 Git（最简单）

```bash
git checkout data_loader.py main.py evaluation.py
```

### 方法 2: 手动恢复

如果你修改了文件，可以重新从 Git 恢复：
```bash
git checkout 5f22083 -- data_loader.py main.py evaluation.py
```

## 📊 运行实验

### 原始实验（同数据集训练和测试）
```bash
# 确保使用原始文件
git checkout data_loader.py main.py evaluation.py

# 运行实验
./run_experiments.sh
```

### 跨数据集实验（在 Scaffold 上训练，在 Size 上测试）
```bash
# 切换到跨数据集版本
cp data_loader_generalization.py data_loader.py
cp main_generalization.py main.py
cp evaluation_generalization.py evaluation.py

# 运行实验
./run_cross_dataset_experiments.sh

# 完成后恢复原始文件
git checkout data_loader.py main.py evaluation.py
```

## ⚠️ 重要提示

1. **运行跨数据集实验前**，务必先切换到 generalization 版本
2. **实验完成后**，记得恢复原始文件，避免混淆
3. **不要删除** `*_generalization.py` 文件，这是你的跨数据集功能备份
4. **修改 ID:OOD 比例**时，编辑当前使用的 `data_loader.py` 文件（第 251 行）

## 🔍 验证当前版本

检查当前使用的是哪个版本：

```bash
# 如果有输出，说明是跨数据集版本
grep -n "test_data_file" main.py

# 如果没有输出，说明是原始版本
```

## 💾 备份说明

所有带 `_generalization` 后缀的文件都是跨数据集功能的备份：
- 包含完整的跨数据集支持
- 已修复缓存 bug
- 支持 --test_data_file 和 --test_drugood_subset 参数

原始文件已从 Git commit `5f22083` 恢复，不包含跨数据集修改。

## 🚀 快速参考

| 操作 | 命令 |
|------|------|
| 切换到跨数据集版本 | `cp *_generalization.py .` (需要正确匹配文件名) |
| 恢复原始版本 | `git checkout data_loader.py main.py evaluation.py` |
| 检查当前版本 | `grep test_data_file main.py` |
| 运行原始实验 | `./run_experiments.sh` |
| 运行跨数据集实验 | `./run_cross_dataset_experiments.sh` (需先切换版本) |
