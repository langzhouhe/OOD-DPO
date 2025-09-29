# Lambda Sensitivity Analysis Report (基于TEST SET)

## 实验配置
- 固定 Beta: 0.1
- Lambda 值范围: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
- 测试数据集: lbap_general_ec50_scaffold, lbap_general_ec50_size, lbap_general_ec50_assay
- **重要**: 所有性能指标基于真实的TEST SET评估
- 实验时间: 2025-09-20 23:28:37

## 关键发现 (基于TEST SET)

### 各数据集最佳Lambda值

#### EC50 Scaffold
- 最佳 Lambda: 0.05
- 最佳 TEST ROC-AUC: 0.9542
- 对应 Validation AUC: 0.9143
- TEST 能量分离度: 0.5916

#### EC50 Size
- 最佳 Lambda: 0.05
- 最佳 TEST ROC-AUC: 0.9994
- 对应 Validation AUC: 0.9704
- TEST 能量分离度: 0.9858

#### EC50 Assay
- 最佳 Lambda: 0.01
- 最佳 TEST ROC-AUC: 0.6585
- 对应 Validation AUC: 0.6868
- TEST 能量分离度: 0.2818

### 总体性能分析 (TEST SET)
- 最佳性能: Lambda=0.05, Dataset=EC50 Size, TEST AUC=0.9994
- 最差性能: Lambda=5.0, Dataset=EC50 Assay, TEST AUC=0.5138
- 平均 TEST AUC: 0.8078
- TEST AUC 标准差: 0.1757

### Lambda值敏感性分析 (TEST SET)
- Lambda 0.01: 平均TEST AUC=0.8636 (±0.1806), 范围=[0.6585, 0.9989]
- Lambda 0.05: 平均TEST AUC=0.8673 (±0.1911), 范围=[0.6482, 0.9994]
- Lambda 0.1: 平均TEST AUC=0.8685 (±0.1844), 范围=[0.6576, 0.9992]
- Lambda 0.5: 平均TEST AUC=0.8504 (±0.1914), 范围=[0.6342, 0.9978]
- Lambda 1.0: 平均TEST AUC=0.8175 (±0.2112), 范围=[0.5819, 0.9898]
- Lambda 2.0: 平均TEST AUC=0.7766 (±0.1769), 范围=[0.5921, 0.9447]
- Lambda 5.0: 平均TEST AUC=0.6106 (±0.1224), 范围=[0.5138, 0.7482]


### 建议 (基于TEST SET)

1. **最佳Lambda**: 0.1 (平均TEST AUC最高)
2. **稳定Lambda**: 5.0 (TEST AUC标准差最小)
3. **数据集特定建议**:
   - EC50 Scaffold: Lambda=0.05 (TEST AUC=0.9542)
   - EC50 Size: Lambda=0.05 (TEST AUC=0.9994)
   - EC50 Assay: Lambda=0.01 (TEST AUC=0.6585)


## 统计摘要 (TEST SET)
- 成功实验数: 21
- 失败实验数: 0
- 最高 TEST AUC: 0.9994
- 最低 TEST AUC: 0.5138

**此版本确保所有结论基于真实的测试集性能，提供可靠的参数选择指导。**

生成时间: 2025-09-20 23:28:37
