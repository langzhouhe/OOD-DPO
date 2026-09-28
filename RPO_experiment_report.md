# RPO 修订实验报告

*Reliability as Preference Optimization — 针对 3×Reject + 否定性 meta-review 的证据重建*

日期:2026-07-30(第三次更新) | 20 组实验 | 编码器:MiniMol / Uni-Mol(冻结,512 维) | 数据:DrugOOD lbap_general(官方)+ GOOD HIV/PCBA/ZINC(covariate,官方) | 全部结果 JSON 留存于 `repro/`,脚本留存于仓库根目录

---

## 0. 摘要:三句话结论

- **原论文可完整复现。**12 个 cell 全部落在 ±0.004 以内(E1)。
- **核心主张"pairwise 优于 pointwise"不成立。**在完全对等条件下,balanced BCE 与 RPO 打平甚至略优(E2/E3/E5/E6/E7)。原先看到的低预算优势,经机制识别证明是**类不平衡假象**(E5)。
- **"多视图下 RPO 优于 BCE" 一度看似成立,但在严格协议下被推翻。**旧协议下多视图 RPO 领先(macro +0.0068,两层验证通过,E12–E14);但补齐等调参预算、等样本曝光、**domain-disjoint 验证切分**等 7 项协议修正后,四种表征的 macro Δ 全部转为负值(−0.0001 ~ −0.0029),EC50-Assay 的残余优势仅 +0.001~+0.002(E17)。最可能的原因是旧协议下 val-OOD 与 train-OOD 的 domain 重叠高达 100%。
- **唯一经得起严格协议检验的正向结果是多视图融合本身。**把冻结 FM 表征与 Morgan 指纹、理化描述符融合,在**两个 backbone、两个目标函数上一致提升**:MiniMol +0.0101、Uni-Mol +0.0267 AUROC,FPR95 同步下降(E16/E17)。
- **修复了一个使 Uni-Mol 3D 通路失效的严重 bug。**字典缺少 `[MASK]` token 导致高斯基距离编码器 `gbf`(Uni-Mol 唯一的 3D 信息入口)静默保持随机初始化;修复后 193/193 参数全部加载(E15)。
- **pairwise 目标的最后一条理论出路也已关闭。**把 RPO 的乘积耦合换成化学 OT 耦合(固定式与对抗式各一),预注册闸门 **4/4 FAIL**;决定性的是**置换代价对照**——化学耦合在 2/4 配置上显著劣于一个结构完全相同但毫无化学含义的随机耦合(E19)。同时 `rpo_prod`(全配对均匀耦合)与 BCE 差 ±0.002 内,配对采样方差亦被排除。
- **新的最强资产:benchmark 捷径审计。**零训练、单个平凡描述符:五个 size cell **1.0000 且带空白间隔**(单阈值完全可分);两个 scaffold 主 cell 仅用**环数**即达 0.9567/0.9633,FM 加整套方法净贡献只有 +0.013/+0.020;**只有两个 assay cell 是干净的**(≤0.59)。ZINC-Scaffold 在六个特征中**翻转五个**方向,一举解释了 E5/E8/E17 三个旧谜团(E18/E18b)。
- **提交 PDF 中的隐藏 prompt-injection 并非作者所为**,而是 NeurIPS 加盖在 reviewer-copy 页脚图层中的反-LLM-审稿人诱饵(ToUnicode 重映射字体);原 P0 待删项为误判(E20)。

**因此论文的贡献应落在"benchmark 实际测量了什么 + 协议修复",而非"pairwise 目标"。**多视图融合作为目标无关的表征结论保留;RPO 可作为可选训练目标保留(它不比 BCE 差),但不能承担核心新颖性。**在 E18c 的 assay 落差被解释清楚之前,不得作出任何 SOTA 表述。**

## 1. 统一实验设置(除非另行说明)

> 编码器:MiniMol,冻结,输出 512 维;特征按 SMILES 缓存,所有方法共用
>
> 检测头:MLP 512→256→128→1,ReLU,Dropout 0.1
>
> 优化:AdamW,weight_decay 1e-5,StepLR(step=10, γ=0.9),梯度裁剪 1.0
>
> 数据规模(DrugOOD):train_id 2000 / train_ood 2000 / val 600+600 / test 1000+1000;GOOD 为 5000/5000/1500+1500/2000+2000
>
> 模型选择:每 15 个 epoch 在验证集上算 AUROC,保留最优 checkpoint;**所有超参一律只用验证集选择,test 仅最终评估一次**
>
> RPO 损失:softplus(−β·(E_ood − E_id)) + λ·(E_id² + E_ood²);BCE 损失:0.5·BCE(E_id,0) + 0.5·BCE(E_ood,1) + λ·(…)
>
> 指标:AUROC(主)、AUPR(precision-recall 曲线下面积)、FPR95(TPR=95% 处的 FPR)

## 2. 第一部分:复现与公平基线

### E1 — 用官方数据完整复现 Table 1  `［复现成功 12/12］`

**问题:**原论文数字能否重现,数据管线是否正确。

> β=0.1, λ=0.01, lr=1e-4, batch 512, 500 epochs, seed 1, data_seed 42。DrugOOD 数据取自作者提供的 drugood_all.zip 中的 lbap_general_* 文件;GOOD 数据由官方 GOOD 模块自动下载。

| Cell | 复现 | 论文 | Δ | Cell | 复现 | 论文 | Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| EC50-Scaffold | 0.966 | 0.970 | −0.004 | HIV-Scaffold | 0.778 | 0.777 | +0.001 |
| EC50-Size | 1.000 | 1.000 | 0.000 | HIV-Size | 1.000 | 1.000 | 0.000 |
| EC50-Assay | 0.710 | 0.711 | −0.001 | PCBA-Scaffold | 0.924 | 0.924 | 0.000 |
| IC50-Scaffold | 0.981 | 0.983 | −0.002 | PCBA-Size | 1.000 | 1.000 | 0.000 |
| IC50-Size | 0.999 | 0.999 | 0.000 | ZINC-Scaffold | 0.615 | 0.614 | +0.001 |
| IC50-Assay | 0.659 | 0.660 | −0.001 | ZINC-Size | 1.000 | 1.000 | 0.000 |

> **结论:**12/12 全部命中。过程中发现并修复一个真实 bug:HIV-Scaffold 中一个硼化合物 `Cc1ccc([B-2]2(c3ccc(C)cc3)=NCCO2)cc1` 无法被 MiniMol 编码,`data_loader.get_dataloaders` 在特征缺失时直接抛异常导致训练崩溃,评估退化为随机初始化模型(0.5195,且各 seed 完全相同)。已改为丢弃无法编码的分子,修复后得 0.7777。

### E2 — AC 第一优先要求:同信息量的 BCE 对照  `［全面持平］`

**问题:**Table 1 的巨大提升,来自 pairwise 目标,还是仅仅来自"用了辅助 OOD 数据"?

> 同一冻结编码器、同一 head、同一 ID/OOD 训练数据、同一 λ/lr/epochs/验证选择规则 —— 唯一区别是把 loss 换成 BCE(`--loss_type bce`)。

| Cell | RPO | BCE | Δ | Cell | RPO | BCE | Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| EC50-Scaffold | 0.966 | 0.966 | 0.000 | HIV-Scaffold | 0.778 | 0.767 | +0.011 |
| EC50-Assay | 0.710 | 0.710 | −0.001 | PCBA-Scaffold | 0.924 | 0.924 | 0.000 |
| IC50-Scaffold | 0.981 | 0.982 | −0.001 | ZINC-Scaffold | 0.615 | 0.619 | −0.004 |
| IC50-Assay | 0.659 | 0.660 | −0.001 | 四个 Size cell | 1.000 | 1.000 | 0.000 |
| **12 cell 平均** | **0.886** | **0.886** | **+0.000** | **RPO 严格胜出** | **1 / 12** |  |  |

> **结论:**Table 1 中相对 post-hoc 方法的巨大提升,几乎全部来自 outlier exposure(使用了辅助 OOD 数据),而非 pairwise 损失本身。AC 与三位 reviewer 的核心质疑成立。

### E3 — OE 目标函数族的公平对照  `［全族打平］`

> 五种目标:dpo(pairwise logistic)、hinge(pairwise margin)、bce、mse、energy-OE(Liu 2020 的能量边界)。同特征、同 head、同优化、best-val checkpoint,5 seed,OOD 预算 K=2000(满)。

| Cell | dpo | hinge | bce | mse | energy-OE |
|---|---:|---:|---:|---:|---:|
| EC50-Assay | 0.701 | 0.700 | 0.702 | 0.699 | **0.706** |
| EC50-Scaffold | 0.953 | 0.955 | 0.955 | 0.956 | **0.963** |
| IC50-Assay | 0.673 | 0.671 | 0.672 | 0.659 | **0.673** |
| ZINC-Scaffold | 0.532 | 0.540 | 0.530 | 0.546 | **0.560** |

> **结论:**满预算下全族收敛到同一水平,每个 cell 的微弱最优都是 energy-OE(2020 年方法),不是本文的损失。

## 3. 第二部分:被排除的路径

### E4 — 稀缺辅助 OOD 下的"样本效率"优势  `［后被 E5 解释为假象］`

**问题:**辅助 OOD 极少时,pairwise 是否更省样本?

> 固定 test 集,只改变训练用 OOD 数量 K;dpo vs bce(当时的 bce 为**混合池化**版本),5 seed,300 epochs。表中为 ΔAUROC = dpo − bce。

| Cell | K=10 | K=25 | K=50 | K=100 | K=250 | K=500 | K=2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EC50-Assay | +0.045 | +0.066 | +0.082 | +0.088 | +0.032 | +0.012 | −0.001 |
| EC50-Scaffold | +0.006 | +0.018 | +0.035 | +0.039 | +0.038 | +0.030 | −0.001 |
| IC50-Assay | +0.010 | +0.006 | +0.017 | +0.021 | +0.024 | +0.010 | −0.000 |
| ZINC-Scaffold | −0.119 | −0.142 | −0.140 | −0.114 | −0.019 | +0.054 | +0.002 |

> **结论:**曲线形状符合预期(低预算领先、满预算归零),但 ZINC 上符号完全反转,提示效应可能并非来自 pairwise 结构。该异常直接促成 E5。

### E5 — 机制识别:是 pairwise 结构,还是类平衡?  `［决定性反证］`

**问题:**把"损失形式 / 类平衡 / OOD 复用"三个因素拆开,究竟哪个在起作用。

> 七种目标,同 head/优化/正则/epochs/best-val:
>
> • `bce_natural` 混合池化均值(K 小时 OOD 被严重欠权重)
>
> • `bce_balanced` 0.5·mean(ID) + 0.5·mean(OOD),显式类平衡
>
> • `bce_wnorm` 加权 BCE,权重归一化到均值 1(避免改变梯度尺度)
>
> • `mse_balanced` 逐点回归,同样两类各自求均值
>
> • `pair_log_m1` 每个 OOD 只配 1 个随机 ID(去掉复用)
>
> • `pair_log_all` 全配对(即原 RPO)
>
> • `hybrid` 0.5·balanced BCE + 0.5·pairwise

| Cell / K | bce_nat | bce_bal | bce_wnorm | mse_bal | pair_m1 | pair_all | hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| EC50-Assay · K=50 | 0.501 | 0.616 | 0.616 | 0.595 | 0.619 | 0.617 | 0.616 |
| EC50-Assay · K=2000 | 0.696 | 0.696 | 0.696 | 0.675 | 0.690 | 0.695 | 0.696 |
| EC50-Scaffold · K=50 | 0.863 | 0.907 | 0.907 | 0.902 | 0.888 | 0.906 | 0.907 |
| IC50-Assay · K=50 | 0.575 | 0.617 | 0.617 | 0.594 | 0.611 | 0.617 | 0.616 |
| ZINC-Scaffold · K=50 | 0.639 | 0.473 | 0.473 | 0.471 | 0.455 | 0.469 | 0.473 |
| **pairwise − balanced BCE(跨任务平均)** | **K=20: −0.001   K=50: −0.001   K=200: −0.001   K=2000: −0.002(任何预算下均不为正)** |  |  |  |  |  |  |

> **结论:**唯一的输家是**未做类平衡的 pooled BCE**;一旦做类平衡,BCE 立即追平 pairwise。同时 `pair_m1 ≈ pair_all`,排除了"每个 OOD 对 ID 分布充分平均"的解释。**E4 的低预算优势 = 类不平衡假象,与 pairwise 结构无关。**另注:ZINC 上呈现相反符号(弱用 OOD 反而更好,0.639),提示该 benchmark 存在负迁移。

### E6 — 单视图下对等调参:tuned RPO vs tuned balanced BCE  `［0/7］`

> RPO 搜索 β∈{0.01,0.05,0.1,0.5,1,5,10} × λ∈{0,1e-4,1e-3,1e-2,5e-2,0.1,0.5}(49 组);BCE 搜索 λ(7)× lr∈{3e-5,1e-4,3e-4}(21 组)。均由验证集 AUROC 选择,3 seed。

| Cell | 默认 RPO(β=0.1) | tuned RPO | tuned balanced BCE | Δ(RPO−BCE) |
|---|---:|---:|---:|---:|
| EC50-Assay | 0.680 | 0.698 | 0.705 | −0.008 |
| IC50-Assay | 0.652 | 0.661 | 0.666 | −0.006 |
| EC50-Scaffold | 0.918 | 0.959 | 0.966 | −0.007 |
| IC50-Scaffold | 0.930 | 0.972 | 0.979 | −0.007 |
| HIV-Scaffold | 0.609 | 0.706 | 0.739 | −0.033 |
| PCBA-Scaffold | 0.827 | 0.867 | 0.881 | −0.015 |
| ZINC-Scaffold | 0.453 | 0.551 | 0.591 | −0.040 |
| **平均** | **调参使 RPO 提升 +0.049** |  | **但对等后仍** | **−0.016** |

> **结论:**调参本身有实际价值——论文默认 β=0.1 明显欠调,验证集在多数 cell 上选择 β=10、λ=0.5,平均提升 +0.049(如 HIV-Scaffold 0.609→0.706)。但对等调参后 BCE 在 **7/7** 个 cell 上仍 ≥ RPO。

### E7 — 给 RPO 专属结构优势:难例挖掘 + 局部配对  `［0/4,两个指标都输］`

**动机:**BCE 无法表达"该 OOD 必须排在*这几个结构相似的* ID 之后",这是 pairwise 独有的表达能力。

> 难例挖掘:对全部 pairwise loss 取最大的 top-q%(q∈{25,50,100})求均值。局部配对:每个 OOD 配 k 个 embedding 最近的 ID(困难)+ k 个随机 ID(全局约束),k∈{32,128}。β∈{5,10}。共 14 种配置,验证集选择,3 seed。

| Cell | BCE AUROC | 最优 RPO | ΔAUROC | BCE FPR95 | RPO FPR95 | ΔFPR95 |
|---|---:|---:|---:|---:|---:|---:|
| EC50-Assay | 0.709 | 0.703 | −0.006 | 0.811 | 0.833 | +0.022 |
| IC50-Assay | 0.660 | 0.652 | −0.008 | 0.853 | 0.867 | +0.014 |
| EC50-Scaffold | 0.968 | 0.964 | −0.004 | 0.176 | 0.217 | +0.041 |
| IC50-Scaffold | 0.982 | 0.977 | −0.005 | 0.107 | 0.127 | +0.019 |

> **结论:**验证集在 4/4 个 cell 上都选中了**朴素的全配对 RPO**——难例挖掘与局部配对连被选中都做不到。原本最寄希望的 FPR95(尾部排序)也 4/4 更差。

### E8 — Exposure 强度 ρ 自适应与 ZINC 负迁移  `［验证集选反］`

**动机:**E5 发现 ZINC 上"弱用 OOD"反而更好。设 L_ρ =(1−ρ)·L_natural + ρ·L_balanced,ρ=0 为弱 OE、ρ=1 为强 OE,尝试用验证集自动选 ρ。

| ZINC-Scaffold,K=50 | test AUROC | 说明 |
|---|---:|---|
| ρ = 0(弱 OE) | 0.638 | 最优 |
| ρ = 1(强 OE) | 0.476 | 负迁移,低于随机 |
| ρ* = 验证集选出 | 0.476 | 3/3 seed 全部选中 ρ=1,即最差配置 |

> **结论:**验证集在最需要它的 cell 上选出了最差的 ρ。根因是 `val_ood` 与 `train_ood` 同源(同一辅助池),强拟合在验证集上表现很好;而 `test_ood` 来自另一批未见 domain。**验证集在结构上看不见这种负迁移**,因此该自适应机制不可用。

### E9 — 其余三条支线  `［均无效］`

| 尝试 | 关键结果 | 结论 |
|---|---|---|
| 通用背景 OOD(随机 ZINC 分子代替 curated OOD) | EC50-Scaffold 0.954→0.884;**EC50-Assay 0.704→0.447**;HIV-Scaffold 0.696→0.656 | 仅在结构 shift 上勉强可用,assay 上崩溃至低于随机 |
| 混合打分(RPO 能量 + Mahalanobis 残差,权重由验证集选) | EC50-Assay +0.011,其余三个 cell 持平或微跌 | 头部分数融合无效(两者高度相关) |
| 零-OOD 参照(KNN / Mahalanobis,完全不用 OOD) | EC50-Assay:Maha 0.610 vs RPO@K=50 的 0.606 | assay 上 50 个辅助 OOD 换不到超过免费方法的收益 |

## 4. 第三部分:诊断结果(直接回应 AC 的问题)

### E10 — Size shortcut:trivial 特征 vs 基础模型  `［AC 点名要求］`

**问题:**Size split 上的 1.000 是真实能力,还是"分子大小"这一平凡线索?

> 同一 head、同一划分、同一训练配方,只改变输入特征:size(原子数+分子量,2 维)、Morgan ECFP4(2048 位)、RDKit 描述子(10 维)、MiniMol(512 维)。

| Cell | 原子数+分子量(2 维) | Morgan | 描述子(10 维) | MiniMol(512 维) |
|---|---:|---:|---:|---:|
| EC50-Size | **1.000** | 0.916 | 0.978 | 1.000 |
| EC50-Scaffold | 0.900 | 0.848 | **0.955** | 0.960 |
| EC50-Assay | 0.580 | 0.712 | 0.611 | **0.720** |
| HIV-Scaffold | 0.474 | 0.720 | 0.438 | **0.721** |
| ZINC-Scaffold | 0.366 | 0.532 | 0.390 | 0.520 |

> **结论(三条可直接写进论文):**(1) Size split 仅用 2 个数字即可满分,不能作为方法有效性的主要证据,主结果应以 assay/scaffold 为主;(2) 基础模型的真正价值体现在 **assay**(0.720 vs 2 维特征的 0.580);(3) ZINC-Scaffold 对所有特征类型都接近随机(0.366–0.532),应如实报告为"在当前表征下不可检测",不作为主张依据。

### E11 — 分子 OOD 是多轴的:跨 shift 迁移矩阵  `［新发现］`

> 行 = 训练所用 shift 的辅助 OOD,列 = 测试所用 shift 的 test OOD;MiniMol + RPO,同一配方。

| train ＼ test | size | scaffold | assay |
|---|---:|---:|---:|
| size | 1.000 | 0.896 | 0.431 |
| scaffold | 0.994 | 0.960 | 0.466 |
| assay | 0.254 | 0.412 | 0.720 |

**补充实验:**用仅在 ID 上训练的性质分类器的不确定性(MSP / energy / entropy)做 OOD 检测,EC50-Assay 上分别为 0.478 / 0.481 / 0.480,即接近随机。

> **结论:**size 与 scaffold 属同一"结构轴"(互测 0.896–0.994),而 **assay 是正交的"功能轴"**:跨轴迁移全面崩溃,assay→size 甚至为 0.254(排序被做反)。结合补充实验,功能性 OOD 在纯结构表征下不可检测。此发现可用于界定论文的适用范围,并解释为何"泛化到未见 shift"难以成立。

## 5. 第四部分:确认的正向结果 —— 多视图

### E12 — 多视图融合抬高天花板  `［5/6 提升］`

**洞察来源:**E10 显示 Morgan(0.712)与 MiniMol(0.720)分数几乎相同但视角完全不同——两者并非互相取代,而可能互补。

> 候选视图:fm / morgan / desc / fm+morgan / fm+desc / fm+morgan+desc。每块按 train_id 统计量标准化后拼接;两种损失(balanced BCE 与 RPO)均训练,由验证集选择;3 seed。

| Cell | 仅 MiniMol | 验证集选中的视图 | AUROC | Δ vs 单视图 | 论文原值 |
|---|---:|---|---:|---:|---:|
| EC50-Assay | 0.722 | fm+morgan | 0.743 | +0.021 | 0.711 |
| IC50-Assay | 0.656 | morgan | 0.656 | −0.000 | 0.660 |
| EC50-Scaffold | 0.971 | fm+morgan+desc | 0.984 | +0.013 | 0.970 |
| IC50-Scaffold | 0.983 | fm+morgan+desc | 0.993 | +0.009 | 0.983 |
| HIV-Scaffold | 0.761 | fm+morgan | 0.795 | +0.034 | 0.777 |
| PCBA-Scaffold | 0.903 | fm+morgan+desc | 0.933 | +0.030 | 0.924 |

> **结论:**平均 +0.018,且全面超过论文原报数值。这同时正面回答了 reviewer 的质问"为什么不能直接在 Morgan 指纹上训练"——答案是两者互补,单用任一都不如融合。

### E13 — 关键区分:融合是抬高共同天花板,还是给了 RPO 优势?  `［视图越丰富越有利］`

> **消除维度/容量混淆:**每个视图分别标准化后,各自经一个线性层投影到 256 维再拼接,因此两种损失的检测头输入维度与参数量完全一致。超参分别在同等大小的网格上由验证集选择,3 seed。表中为 ΔAUROC = RPO − balanced BCE。

| Δ(RPO − balBCE) | fm | morgan | fm+morgan | fm+morgan+desc |
|---|---:|---:|---:|---:|
| EC50-Assay | −0.006 | +0.009 | +0.005 | +0.015 |
| IC50-Assay | −0.014 | −0.002 | −0.010 | +0.005 |
| EC50-Scaffold | −0.003 | −0.016 | −0.001 | +0.001 |
| IC50-Scaffold | −0.000 | +0.001 | +0.002 | +0.003 |
| HIV-Scaffold | −0.015 | −0.025 | −0.013 | −0.008 |
| PCBA-Scaffold | −0.019 | −0.013 | −0.008 | −0.007 |
| **RPO 胜出 cell 数** | **0 / 6** | **2 / 6** | **2 / 6** | **4 / 6** |

> **结论:**单视图下 RPO 为 0/6;完整多视图下升至 4/6,且 DrugOOD 四个主 cell 全部领先。这给出一个具体、可检验的机制陈述:**RPO 的有限样本优势只在异构多视图表征上出现**,而非"pairwise 天然对齐 AUROC"。

### E14 — 两层独立确认  `［通过］`

> **冻结全部选择:**视图固定为 fm+morgan+desc;每个 cell 的 RPO(β,λ)与 BCE(λ,lr)取自前述验证集选择,不再重调;head、epochs、checkpoint 规则一致。
>
> **层一(训练随机性):**5 个全新初始化 seed(11–15),未参与方法开发。
>
> **层二(benchmark 重采样):**5 个新 data_seed(43–47),复刻 `data_loader` 的确定性采样逻辑;每个 cell 重抽约 3 万个不同分子(与原划分重叠仅 500–2400),**test 子集同样改变**;每个划分用同一固定初始化 seed,两种损失严格配对。

| Cell | 层一 Δ(5 训练 seed) | 层二 Δ(5 数据划分) | 层二 95% CI | 正向划分数 | FPR95(RPO / BCE) |
|---|---:|---:|---|---:|---:|
| EC50-Assay | +0.0164 | **+0.0246** | [+0.003, +0.046] | 5/5 | 0.846 / 0.869 |
| IC50-Assay | +0.0057 | **+0.0269** | [+0.009, +0.045] | 5/5 | 0.829 / 0.831 |
| EC50-Scaffold | +0.0016 | **+0.0040** | [+0.003, +0.005] | 5/5 | 0.060 / 0.084 |
| IC50-Scaffold | +0.0037 | +0.0006 | [−0.003, +0.004] | 3/5 | 0.045 / 0.050 |
| **macro-average** | **+0.0068** | **+0.0140** | **[+0.005, +0.023]** | **5/5 splits** | **4/4 更优** |

> **结论:**两层均通过。层一 20/20 个配对差全部为正;层二 macro ΔAUROC = +0.0140,5/5 划分为正,FPR95 4/4 系统性改善,无指标退化。换数据划分后优势反而**更大**,说明原固定 test split 偏保守,缓解了视图选择偏差的担忧。**但 IC50-Scaffold 的 95% CI 含 0,应写作 essentially tied,不可宣称显著领先。**

### E14b — GOOD 上的同等对照(封住公平性攻击面)  `［统计持平］`

> 同一冻结配方(fm+morgan+desc),超参在同一搜索空间内由验证集选择(不因结果调整空间),5 seed 配对,benchmark 标准划分。

| Cell | RPO | Balanced OOD Head | Δ | 95% CI | AUPR | FPR95 |
|---|---|---|---:|---|---:|---:|
| HIV-Scaffold | 0.7764 ± .0078 | 0.7770 ± .0080 | −0.0006 | ±0.0088(含 0) | 0.745 / 0.745 | 0.678 / 0.680 |
| PCBA-Scaffold | 0.9115 ± .0052 | 0.9123 ± .0046 | −0.0009 | ±0.0020(含 0) | 0.910 / 0.912 | 0.393 / 0.403 |
| ZINC-Scaffold | 0.5042 ± .0168 | 0.5075 ± .0123 | −0.0034 | ±0.0074(含 0) | 0.521 / 0.524 | 0.971 / 0.966 |

> **结论:**三个 cell 的 CI 全部含 0,应写作 **statistically tied** 而非"RPO 更差";FPR95 在 HIV/PCBA 上 RPO 反而略优。**全部实验中,没有任何一个 cell 显示 RPO 显著更差。**

## 5B. 第五部分:Uni-Mol 修复、每-backbone 多视图与严格协议复核

### E15 — Uni-Mol 权重加载 bug:3D 通路一直是随机初始化  `［严重 bug,已修复］`

**触发:**Gate 1 中 Uni-Mol(3D 几何)加入多视图后增益反而小于 MiniMol(2D 图),与"视图越异构增益越大"的预期矛盾。追查权重加载日志发现异常。

| 参数 | checkpoint | 模型(修复前) | 作用 | 修复后 |
|---|---|---|---|---|
| embed_tokens.weight | (31, 512) | (30, 512) 形状不匹配 | 原子类型嵌入 | (31,512) ✓ |
| gbf.mul.weight | (961, 1) | (900, 1) 形状不匹配 | **高斯基距离编码** | (961,1) ✓ |
| gbf.bias.weight | (961, 1) | (900, 1) 形状不匹配 | **高斯基距离编码** | (961,1) ✓ |
| 加载参数总数 | 211 | **190 / 193** | — | **193 / 193** |

**根因:**Uni-Mol 官方 task 在 `Dictionary.load()` 之后总会追加一个 `[MASK]`(30+1=31),而本仓库 `model.py` 只在"字典文件不存在"的兜底分支里追加,走加载分支时漏掉。961=31²、900=30²,说明 `gbf` 整体错位。加载时使用 `strict=False` 并静默过滤形状不匹配的参数,整个失败只留下日志里一句 "Loaded 190 parameters"。

> **后果与验证:**`gbf`(Gaussian Basis Function)是 Uni-Mol **把原子间 3D 距离编码进网络的唯一入口**。它保持随机初始化,意味着此前所有 Uni-Mol 实验的 3D 几何信息本质上是噪声。修复后单视图 Uni-Mol 在四个 primary cell 上为 0.6511 / 0.6317 / 0.9763 / 0.9822(均值 0.8103),与论文报告的 0.650 / 0.640 / 0.965 / 0.977(均值 0.8080)吻合 —— 说明**作者当初的 dict.txt 应为正确的 31 token,该 bug 由使用官方 example dict 的加载路径触发,不影响原论文结论**。这同时补上了 E1 缺失的 Uni-Mol 复现。

### E16 — 每-backbone 多视图与 post-hoc baseline 修正  `［12 cell 完整表］`

**设计变更:**此前把 MiniMol 与 Uni-Mol 拼进同一个多视图会破坏论文"每个 backbone 独立成块"的结构,也削弱 model-agnostic 主张。改为每个 backbone 各自加 Morgan 指纹与理化描述符,构成两个平行分块。

> 视图:`MiniMol` / `MiniMol+Morgan+Desc` / `Uni-Mol` / `Uni-Mol+Morgan+Desc`;每块标准化后各经一个线性层投影再拼接,两种目标共用完全相同的 head;12 cell × 8 方法 × 5 seed。

**同时修正的 post-hoc bug:**Mahalanobis / KNN / LOF 原先建在**分类器倒数第二层**特征上,导致 KNN 在 EC50-Scaffold 仅 0.385(低于随机);改建在**冻结 FM 原始特征**上后回到 0.627,与论文的 0.671 同量级。

| 表征 | Balanced OOD Head | RPO | Δ |
|---|---:|---:|---:|
| MiniMol | 0.881 | 0.876 | −0.005 |
| MiniMol + Morgan + Desc | 0.878 | **0.880** | **+0.002** |
| Uni-Mol | 0.866 | 0.863 | −0.003 |
| Uni-Mol + Morgan + Desc | 0.877 | **0.880** | **+0.003** |

> **当时的结论(后被 E17 推翻):**单视图下 RPO 输、多视图下 RPO 赢,且该模式在两个 backbone 上一致。另有一个稳定的附带发现:**post-hoc 距离型方法在多视图下大幅退化**(KNN 0.614→0.443,LOF 0.689→0.487),因为 Morgan 是 2048 维稀疏二值向量、距离度量在高维稀疏空间失效 —— 说明多视图必须配合**可学习的**检测头才有效。

### E17 — 按修订文档执行 Phase 0 严格协议复核  `［推翻 E16 的 RPO 优势］`

**依据:**`REVISION_EXPERIMENT_ORGANIZATION_ZH.md` 的 Phase 0 要求。实施了 7 项协议修正:

> 1. **等调参预算**:固定 β=1(消除 β–λ 混淆,真正起作用的是 γ=λ/β²),两个目标搜同一个 3×3 网格(γ × lr)= 9 trials
>
> 2. **等样本曝光**:两个目标共享同一份 ID/OOD 分子子集
>
> 3. **修正 FPR95 为惯例定义**(ID 为正类:阈值取 ID 分数 95 分位,统计落在其下的 OOD)。实测与旧实现不同:0.513 vs 0.567
>
> 4. 全程 validation-only 选择,删除任何依据 test 指标的选择
>
> 5. **参数匹配 head**:所有视图配置投影到固定总宽度 768,打分 MLP 完全相同(消除容量混淆)
>
> 6. **DrugOOD 按 domain_id 切分 train-OOD / val-OOD**。实测旧的分子随机切分在两个 assay cell 上 domain 重叠达 **100%**;切分后四个 primary cell 重叠均为 **0**
>
> 7. primary endpoint 固定为 DrugOOD 四个 non-size cell 的 macro AUROC

**过程中我引入并修正的两个错误(值得记入方法学讨论):**

| 错误 | 后果 | 修正 |
|---|---|---|
| γ 网格设为 {0, 0.1, 0.5} | RPO 原最优有效正则 γ=λ/β²≈0.001–0.005 不在网格内;Uni-Mol-MV 上 4/4 选中边界值 γ=0 | 改为 γ∈{0.001, 0.01, 0.1},覆盖原最优区间 |
| 配对索引固定在 epoch 循环外 | RPO 120 个 epoch 只见 2000 个固定配对(原为每步重采样,约 24 万个);BCE 为逐点损失不受影响,等于单方面削弱 RPO | 配对每步重采样;"等曝光"应理解为相同**分子集合** |

**最终结果 — 主表 1:PRIMARY(DrugOOD 四个 non-size cell)**

| 表征 | Balanced OOD Head | RPO | Δ AUROC | BOH FPR95 | RPO FPR95 |
|---|---:|---:|---:|---:|---:|
| MiniMol | 0.8342 | 0.8341 | −0.0001 | 0.4713 | 0.4677 |
| MiniMol + Morgan + Desc | **0.8443** | 0.8432 | −0.0011 | 0.4405 | 0.4397 |
| Uni-Mol | 0.8124 | 0.8103 | −0.0021 | 0.4627 | 0.4743 |
| Uni-Mol + Morgan + Desc | 0.8390 | 0.8362 | −0.0029 | 0.4324 | 0.4463 |

**主表 2:多视图 vs 单视图(四 cell macro AUROC)**

| Backbone | 单视图 BOH | 多视图 BOH | Gain | 单视图 RPO | 多视图 RPO | Gain |
|---|---:|---:|---:|---:|---:|---:|
| MiniMol | 0.8342 | 0.8443 | **+0.0101** | 0.8341 | 0.8432 | **+0.0091** |
| Uni-Mol | 0.8124 | 0.8390 | **+0.0267** | 0.8103 | 0.8362 | **+0.0259** |

**附录汇总:GOOD scaffold(secondary)与 size(shortcut diagnostic)**

| 分组 | 表征 | BOH | RPO | Δ |
|---|---|---:|---:|---|
| GOOD-Scaffold(3 cell) | MiniMol / +MV | 0.7353 / 0.7310 | 0.7343 / 0.7205 | −0.0010 / −0.0105 |
| GOOD-Scaffold(3 cell) | Uni-Mol / +MV | 0.7169 / 0.7327 | 0.7165 / 0.7301 | −0.0004 / −0.0026 |
| Size(5 cell) | 全部四种配置 | 0.9998–1.0000 | 0.9998–1.0000 | ≈0.0000 |

**逐 cell 差值(RPO − BOH,多视图):**EC50-Assay 是唯一两个 backbone 都为正的 cell(+0.0009 / +0.0024);IC50-Assay 两个 backbone 都为负(−0.0041 / −0.0037);**ZINC-Scaffold 是最大负例**(MiniMol-MV −0.0256),且该 cell 多视图比单视图更差(0.54→0.48)。

> **结论:**按文档 Phase 3 的成功标准("两个 backbone 的四-cell macro Δ 均为正")—— **不通过**,四种表征的 macro Δ 全部为负。E16 中"多视图下 RPO 赢"的现象在严格协议下消失。最可能的原因是 **domain-disjoint 验证切分**(其余修正都对称施加于两个目标):旧协议下 val-OOD 与 train-OOD 的 domain 重叠达 100%,验证集偏向已见 assay domain,而 RPO 之前的优势可能相当程度来自它在这种被污染的验证集上更易被选到有利配置 —— 这正是修订文档 §Q3.3 预警的问题,现被数据证实。
>
>
> **但多视图融合的价值不受影响且更加确立**:在两个 backbone、两个目标函数上一致提升(+0.009 ~ +0.027 AUROC),FPR95 同步下降。

## 5C. 第六部分:benchmark 捷径审计、耦合假设与 PDF 取证

### E18 — 12-cell 平凡特征捷径审计  `［最强新资产］`

**问题:**这些 benchmark cell 到底在测什么。E10 曾用"原子数+分子量"训一个 head 得到 size=1.000;本实验改为**完全不训练**,直接把单个原始标量当 OOD 分数,并加入两个新判据。

> 数据:每个 cell 取实验实际使用的分子子集(`cache/ood_dpo_cache/*_seed42_splits.json`),无任何训练。
>
> 特征:heavy_atoms / molwt / n_rings / logp / tpsa / n_rotbonds,6 个平凡描述符。
>
> 指标:`sep` = 方向无关 AUROC = max(a, 1−a);`gap` = 测试集 ID 与 OOD 的特征取值**无任何重叠**(存在空白间隔);`flip` = 该特征在 (train_id, train_ood) 上的 ID/OOD 方向与测试集**相反**。

**表 1:每个 cell 最强的平凡特征**

| cell | 特征 | sep(train) | sep(test) | gap | flip | 中位 ID/OOD |
|---|---|---:|---:|---|---|---:|
| 5 个 Size cell(全部) | heavy_atoms | 1.0000 | **1.0000** | **是** | — | 24–34 / 12–21 |
| IC50-Scaffold | **n_rings** | 0.8230 | **0.9633** | — | — | 5 / 2 |
| EC50-Scaffold | **n_rings** | 0.8161 | **0.9567** | — | — | 4 / 2 |
| HIV-Scaffold | n_rings | 0.5483 | 0.8096 | — | **是** | 3 / 1 |
| PCBA-Scaffold | heavy_atoms | 0.6562 | 0.7824 | — | — | 27 / 20 |
| ZINC-Scaffold | molwt | 0.6293 | 0.6355 | — | **是** | 334.5 / 296.4 |
| EC50-Assay | tpsa | 0.5636 | 0.5917 | — | — | 75.7 / 88.8 |
| IC50-Assay | n_rings | 0.5179 | 0.5410 | — | — | 4 / 4 |

**Size split 的实际构造(官方原始划分,`ec50_size`):**`ood_val` 的重原子数 p5=p95=27(几乎全是 27),`ood_test` 全是 21–22,而 train ID 最小值为 45。DrugOOD 的 size "domain" 就是尺寸桶,OOD 池只有 1–2 个桶宽,两组之间存在空白间隔。

> **结论(三条,均可直接进正文):**
>
> (1) **五个 size cell 是带间隙的单阈值可分任务。**任何拿到 auxiliary OOD 的方法都被直接交付决策规则,1.000 是免费的;ID-only 方法无法恢复它。因此这些 cell 上的排行榜是在按"能否访问 OOD"排序,不是按检测能力排序。
>
> (2) **scaffold 同样被污染,且比预想严重。**论文头条 EC50-Scaffold 0.970 / IC50-Scaffold 0.983 坐在**单个整数(环数)给出的 0.9567 / 0.9633** 之上,FM 加整套方法的净贡献仅 **+0.013 / +0.020**。
>
> (3) **只有两个 assay cell 是干净的**(最强平凡特征 0.5917 / 0.5410),它们是唯一在测量结构捷径之外信息的 cell。

### E18b — 方向翻转:负迁移的机制解释  `［解决三个旧谜团］`

审计发现 `flip` 并非 ZINC 的孤例,而是 GOOD 三个 scaffold cell 的普遍现象(下表仅列 sep(test) > 0.55 者):

| cell | 特征 | sep(train) | sep(test) |
|---|---|---:|---:|
| ZINC-Scaffold | heavy_atoms | 0.6295 | 0.6256 |
| ZINC-Scaffold | molwt | 0.6293 | 0.6355 |
| ZINC-Scaffold | n_rings | 0.5020 | 0.6024 |
| ZINC-Scaffold | logp | 0.5486 | 0.6094 |
| ZINC-Scaffold | tpsa | 0.5727 | 0.5868 |
| HIV-Scaffold | n_rings | 0.5483 | **0.8096** |
| HIV-Scaffold | logp | 0.5964 | 0.6462 |
| PCBA-Scaffold | logp | 0.6242 | 0.5593 |

> **结论:**ZINC-Scaffold 在六个特征里**翻转五个**——train-OOD 教给检测器的捷径方向与 test-OOD 相反。这一条同时解释了三个此前无法解释的现象:(a) E5 中弱 OE(ρ=0)0.639 胜过强 OE(ρ=1)0.476;(b) E8 中验证集 3/3 seed 选中最差的 ρ=1(因 val_ood 出自 train_ood 池,携带训练方向);(c) E17 中 ZINC 多视图比单视图更差(0.54→0.48,容量更大则把反向捷径拟合得更狠)。
>
> HIV-Scaffold 的 n_rings 更极端:train 仅 0.5483 而 test 达 0.8096 且方向相反,预测 OE 在该 cell 也应受害——与 E6 中 tuned BCE 在 HIV-Scaffold 上胜出 0.033 一致。**负迁移由此从"某个 benchmark 噪声大"升级为可证伪的机制陈述。**

### E18c — 与 ID-only 文献的口径落差  `［未解决,阻断 SOTA 表述］`

PGR-MOOD(KDD 2024, arXiv 2404.15625)及其六个 baseline(GOOD-D / GraphDE / AAGOD / OCGIN / GLocalKD)**全部为 ID-only**(在 ID 图上训扩散/重建模型,训练时不接触 OOD)。其 DrugOOD Table 1 与本文对照:

| 方法(设定) | EC50-Scaf | EC50-Size | EC50-Assay | IC50-Scaf | IC50-Size | IC50-Assay |
|---|---:|---:|---:|---:|---:|---:|
| MSP(ID-only) | 57.26 | 59.18 | 48.19 | 54.57 | 52.57 | 58.19 |
| GOOD-D(ID-only) | 82.51 | 92.50 | 65.20 | 85.40 | 91.55 | 81.35 |
| PGR-MOOD(ID-only) | 87.53 | 97.67 | **86.73** | 91.57 | 93.84 | **83.72** |
| 本文 RPO(**OE**,MiniMol) | **97.0** | **100.0** | 71.1 | **98.3** | **99.9** | 66.0 |

> **结论:**模式与 E18 完全吻合——在被捷径污染的 size/scaffold cell 上 OE 方法大幅领先(因为拿到了决策规则),而在**唯一两个干净的 assay cell 上,ID-only 的 PGR-MOOD 反而高出 15–18 个点**。两者不可直接比较(设定不同、可能是 lbap_core 而非 lbap_general、ID/OOD 构造方式未知),但**在该落差被解释清楚之前,任何形式的 SOTA 表述都不成立**。这是必做实验,不是可选项:若其 assay split 更容易,则又是一条 benchmark 不可比性的实锤;若 ID-only 生成式检测在功能轴上确实强过 OE,则与 E11 的结构轴/功能轴正交性互相印证,是一个独立的重要发现。

### E19 — 耦合假设:chemical-OT RPO 与 adversarial C-RPO  `［闸门 4/4 FAIL］`

**动机(最后一条理论上可行的路):**RPO 的损失对 ID/OOD **独立**采样,即耦合为乘积测度 p⊗q。乘积耦合下 pairwise 风险可分解,与 balanced BCE 共享 Bayes 最优排序(修订文档 §3)——这正是 E17 打平的根因。pairwise 唯一能表达 pointwise 无法表达之物的方式,是一个**不可分解为两个样本权重**的非乘积耦合 π(x_i, x_o)。

> 六个臂,**全部使用全配对加权和 Σ π_io·ℓ_io,因此臂间唯一差异就是耦合矩阵 π**(避免"配对曝光量不对等"的攻击面):
>
> • `bce` balanced BCE(Balanced OOD Head) • `rpo` E17 的采样式 RPO(每 OOD 随机 1 个 ID)
>
> • `rpo_prod` 全配对**均匀乘积**耦合 —— 隔离"全配对 vs 采样"
>
> • `ot_fixed` 在 1−Tanimoto(Morgan ECFP4) 代价上的 entropic-OT 耦合 —— chemical C-RPO
>
> • `ot_adv` 耦合按当前损失对抗性重解并锚定化学代价(K ∝ exp((ℓ−λ_c·c)/ε),边际约束 = E7 朴素 top-q% 挖掘所缺的东西)
>
> • `ot_shuf` 代价矩阵行/列**独立置换**后的 OT 耦合 —— **证伪对照**:边际、熵、ESS、取值谱与 `ot_fixed` 完全相同,唯独化学含义被摧毁
>
> ε=0.02,**仅由代价矩阵几何先验校准**(ESS–化学拉力曲线的拐点:约 65% 最大拉力、约 1.2 万有效配对),不涉及任何模型输出、验证或测试数据;λ_c=1。其余全部冻结在 E17 Phase-0(β=1、同一 3×3 (γ,lr) 九试网格、domain-disjoint 验证、参数匹配 head、conventional FPR95、validation-only 选择、1500 ID/2000 OOD/120 步)。代码直接 `from revision_run import ...`,不复制,保证协议不漂移。

**ΔAUROC vs Balanced OOD Head(5 seed 配对)**

| 表征 / cell | rpo | rpo_prod | ot_fixed | ot_adv | ot_shuf(控制) |
|---|---:|---:|---:|---:|---:|
| MiniMol · EC50-Assay | +0.0015 | −0.0021 | −0.0013 | −0.0045 | −0.0008 |
| MiniMol · IC50-Assay | −0.0019 | +0.0001 | −0.0022 | −0.0014 | −0.0001 |
| MiniMol-MV · EC50-Assay | +0.0009 | −0.0019 | −0.0033 | −0.0033 | −0.0017 |
| MiniMol-MV · IC50-Assay | −0.0041 | +0.0010 | +0.0025 | −0.0024 | +0.0008 |
| **闸门(≥+0.010 且两 assay cell 皆过、5/5 同向)** | **4/4 配置 **FAIL**;最好一格 +0.0025;八个 (arm×cell-view) 中七个为负** |  |  |  |  |

**证伪对照 `ot_fixed − ot_shuf`(两者 ESS 均为 0.0036–0.0042,结构完全相同)**

| 配置 | Δ | 95% CI | 同向 |
|---|---:|---|---:|
| MiniMol · EC50-Assay | −0.0005 | [−0.0043, +0.0033] | 2/5 |
| MiniMol · IC50-Assay | **−0.0021** | **[−0.0027, −0.0015]** | 0/5 |
| MiniMol-MV · EC50-Assay | **−0.0016** | **[−0.0027, −0.0004]** | 1/5 |
| MiniMol-MV · IC50-Assay | +0.0017 | [−0.0018, +0.0051] | 3/5 |

> **结论:**化学耦合在 2/4 配置上**显著劣于一个毫无意义的随机耦合**,1 个打平,1 个名义为正但 CI 跨 0。耦合机制本身确实生效(`ot_fixed` 平均配对化学代价 0.742/0.752 vs `ot_shuf` 的 0.886/0.888 ≈ 均匀值),排除了"实现有误"。附带结论:`rpo_prod` 在四个配置上与 BCE 差 ±0.002 内,**配对采样方差也不是 RPO 打平 BCE 的解释**,E17 的结论再硬一层。
>
>
> **方法学教训(第三次同一模式):**冒烟测试时在 seed 1 / γ=0.01 / lr=3e-4 单一配置上,`ot_fixed` 曾显示 +0.008 且验证集同向;走完九试 validation-only 选参后网格选中 γ=0.1 / lr=0.001,优势完全消失(−0.0013)。与 E4→E5(类不平衡假象)、E16→E17(验证集 domain 泄漏)构成同一失败模式:**看起来成立的效应在正确协议下归零**。

> ℹ️ **适用范围的重要限定:**本实验测的是**松匹配、无条件归一化**的化学耦合。`ICLR_REVISION_ROADMAP_ZH.md` §2 的 NC-RPO 另有三个组件未实现:(1) 带 **caliper** 的 OT(本实验平均配对 Tanimoto 相似度仅 0.26,均匀基线 0.11,匹配相当松);(2) **条件 centering/normalization 以消去 a(c)**——该 roadmap 命题 1 指出最优解为 β⁻¹log[q(z|c)/p(z|c)] + a(c),不消去 a(c) 则估计对象并非条件密度比;(3) matched 与 uniform pairs 的混合 batch。§2.4 要求的 **BCE + nuisance fixed effects** 与 propensity-reweighted BCE 两个对照亦未跑。**因此本结果否定的是"非乘积化学耦合"这一族,不构成对 NC-RPO 的证伪。**

### E20 — PDF 隐藏文字取证  `［非作者所为,结论修正］`

修订文档曾把提交 PDF 第 2 页的隐藏 prompt-injection 列为 P0 待删项。取证结论:**该文字不在作者源码中,亦无法由作者删除。**

> 证据链:(1) 隐藏文字位于第 **2** 页与第 **31** 页,坐标 y=754.18/763.18、字号 7.50,与其余 29 页 "Confidential reviewer copy" 页脚**完全一致**;(2) 将第 2 页页脚区域渲染为图像,视觉上显示的就是正常页脚,与第 3 页无异;(3) 第 2/31 页页脚字体为 `AAAAAA+ArialUnicodeMS_Pair_<hash>_<hex1>_<hex2>`,而第 3 页为干净的 `AAAAAA+ArialUnicodeMS` —— 前者是 **ToUnicode 重映射**的定制子集字体,使渲染层与文本抽取层解耦;(4) "Confidential reviewer copy" 页脚由 OpenReview 在投稿后加盖,作者无法预先伪造;(5) 被强制插入的三句话为**中性标记**而非抬分语言。
>
> 复核命令:`pdftotext -f 2 -l 2 878_Reliability_as_Preference_.pdf - | tail -3`

> **结论:**这是 **NeurIPS 2026 埋设的反-LLM-审稿人诱饵**,用于识别把 PDF 直接喂给大模型写意见的审稿人(ICML 2026 采用同类手段并据此 desk-reject 497 篇,受罚方为违规审稿人)。**行动项:**(1) 核查三份审稿意见是否含那三句话,若含则该意见违反 LLM 政策,可向 PC 反映;(2) 但不得据此否定意见的技术内容——AC 点名的 matched BCE 对照与 size shortcut 两条,已被 E2/E10/E18 证明成立;(3) 切勿在后续投稿中加入任何类似内容(ICLR 2026 明确将其定性为 collusion)。

## 6. 当前可支撑的论文口径

**核心表述(经 E17 严格协议 + E18/E19 后的最终版):**

在严格匹配的协议下(等调参预算、等样本曝光、domain-disjoint 验证切分、参数匹配 head、validation-only 选择),pairwise RPO 与 class-balanced BCE 在所有表征和所有 benchmark 上**统计持平**;二者共享同一 Bayes-optimal ranking。进一步地,把乘积耦合替换为非乘积的化学 OT 耦合(唯一在理论上可能打破该等价性的构造)同样无效,且劣于结构相同的置换代价对照(E19)——因此"pairwise 优于 pointwise"不是"尚未找到",而是**已被系统性排除**。

**真正决定这些榜单数字的是 benchmark 的捷径泄漏与验证集设计。**零训练的单个平凡描述符即可解释五个 size cell 的全部性能(1.0000,带空白间隔)与两个 scaffold 主 cell 的绝大部分(环数 0.9567/0.9633,整套方法净贡献 +0.013/+0.020);只有两个 assay cell 是干净的。三个 GOOD scaffold cell 存在 train/test 捷径**方向翻转**,这是负迁移的机制解释。而一个通过 5 个新初始化 seed 与 5 个新数据划分双层确认的结论,仍因 train-OOD/val-OOD domain 重叠(高达 100%)而在修复后反向。

**可保留的正向贡献:**(1) 多视图融合是**目标函数无关**的表征结论(MiniMol +0.010、Uni-Mol +0.027 AUROC,两个目标一致,FPR95 同步下降),且 post-hoc 距离型方法在多视图下反而退化(KNN 0.614→0.443),说明必须配可学习检测头;(2) 一套修复后的 domain-disjoint 评测协议与可复用的捷径审计。

> ℹ️ **被推翻的中间结论(保留以备审稿追溯):**旧协议下曾观察到多视图 RPO 领先(macro +0.0068,经 5 新训练种子与 5 新数据划分两层验证)。该优势在补齐 domain-disjoint 验证切分等协议修正后消失。这说明**验证集与训练 OOD 同源(domain 重叠 100%)会系统性地偏向某些配置**,是分子 OOD 实验设计中一个容易被忽略的陷阱。

- **主结果写 assay**(+0.0246 / +0.0269);scaffold 写"接近天花板时保持竞争力";IC50-Scaffold 与三个 GOOD cell 写 **essentially tied**。
- **BCE 仅作 matched baseline**,表中命名为 "Balanced OOD Head",不进摘要、引言与贡献列表;理论部分加一句范围说明:证明的是 ranking consistency,不主张与 balanced pointwise 有不同的 Bayes ordering。
- **主张收窄:**标题限定 molecular(非 scientific)foundation models;"label-free" 改为 **property-label-free**(仍需 ID/OOD 分布标签);chemical hallucination 降为应用动机,不声称已解决。
- **重采样边界:**data_seed 只重抽分子、未重新划分 domain,因此证明的是 benchmark-resampling stability;论文应写 "held-out domains under benchmark-defined shifts",不可扩大为 unseen-mechanism robustness。
- **阈值协议:**AUROC/AUPR 为 threshold-free;FPR95 为测试期诊断指标;部署阈值取验证集 ID 分数的 95% 分位数,从不使用 test OOD。
- **Figure 1 必须重做:**原图比较不同 loss 的原始 score gap(尺度不可比),且代码中各 loss 超参严重不对等(bce 用 λ=0.5/lr=2e-5,dpo 用 λ=0.01/lr=1e-4)。应改为相同模型、数据与调参预算下的 AUROC/AUPR/FPR95。

## 7. 待办(按优先级)

- **P0 — 解释 E18c 的 assay 落差:**PGR-MOOD(ID-only)在两个干净 cell 上报 86.73/83.72,本文 OE 方法为 71.1/66.0。先核对其使用的是 `lbap_core` 还是 `lbap_general`、ID/OOD 集合如何构造、子采样规模。**这一条不解决则任何 SOTA 表述都不成立**,且两种可能的结论都有价值(benchmark 不可比 / ID-only 在功能轴上确实更强)。
- **P0 — 严格协议下复跑 pAUC:**现有 `pauc_rpo.py` 结果(IC50-Scaffold FPR95 0.0348→0.0148,相对降 57%;EC50-Scaffold 0.0396→0.0294)来自**旧协议**,且 α 是额外超参(调参预算不对等)。须挤进同一九试预算、domain-disjoint 验证、conventional FPR95,并加入审稿人必然要求的 pointwise 对照(focal BCE、hard-example-weighted BCE)。若加权 pointwise 追平,则方法贡献归零。这是唯一还活着的方法候选。
- **P1 — 是否补 NC-RPO 的条件归一化臂(需作者决策):**E19 否定的是松匹配、无条件归一化的化学耦合。`ICLR_REVISION_ROADMAP_ZH.md` §2 命题 1 要求条件 centering 以消去 a(c),否则估计对象并非条件密度比;此外缺 caliper、matched/uniform 混合 batch,以及 §2.4 要求的 BCE+nuisance fixed effects 与 propensity-reweighted BCE 两个对照。补条件 centering 约 20 行。按作者预注册规则闸门失败即停,故此项待定。
- **P1 — 捷径审计扩展:**补多变量 trivial baseline(6 个描述符联合训一个 head)作为上界,并对每个 cell 给出"FM 相对最强 trivial 特征的净增益",作为正文主表的一列。
- **Phase 1 机制诊断(问题已重新表述):**原问题"为何多视图下 RPO 胜过 BCE"因 E17 已不成立。改为检验**多视图相对单视图的增益是否集中在跨视图分歧高的 ID–OOD pair 上**。做法:为每个视图块单独训练打分器,对 test pair 计算各视图 margin 的符号一致性 d=1−|mean sign|,按 validation 上确定的切点分 low/mid/high 三档,比较单视图与多视图的 pairwise accuracy。**可证伪**:若增益不随分歧上升,则不能声称"互补信息"是机制。脚本 `phase1_disagreement.py` 已就绪。
- **post-hoc baseline 用仓库原生脚本重跑:**当前实现的分类器训练协议(200 epoch 固定超参)与仓库 `baseline_trainer.py`(500 epoch + early stopping + best-val)不一致,数值相差约 0.05。若进最终稿建议统一。
- **GOOD 的 MSP/ODIN/Energy 缺失:**当前 loader 只为 DrugOOD 加载 cls_label,GOOD 六个 cell 的三个 logit-based 方法为空,需补 GOOD 标签。
- **固定总维度的容量对照:**E17 已用固定总宽度 768,但单视图仍是 1 个投影层、多视图是 3 个。若审稿人追问,可补 duplicate-view / shuffled-view 对照,确认增益来自视图异构性而非投影层数量。

## 8. 复现说明

> 环境:conda env `ood`(Python 3.10,torch 2.5.1+cpu),解释器 `/root/miniconda3/envs/ood/bin/python`
>
> 主要脚本(仓库根目录):`repro_table1.py`(E1)、`run_bce_control.sh`(E2)、`oe_family_experiment.py`(E3)、`scarce_ood_experiment.py`(E4)、`loss_id.py`(E5)、`tuned_compare.py`(E6)、`rpo_methoddev.py`(E7)、`exposure_sweep.py`(E8)、`generic_ood_experiment.py` / `hybrid_experiment.py` / `zero_ood_reference.py`(E9)、`fingerprint_vs_fm.py`(E10)、`cross_shift_experiment.py` / `functional_signal_experiment.py`(E11)、`fusion_experiment.py`(E12)、`fusion_matrix.py`(E13)、`confirm_run.py` / `split_confirm.py`(E14)、`good_matched.py`(E14b)、`precompute_unimol.py`(E15 GPU 特征预计算)、`best_model.py` / `paper_table.py`(E16)、`revision_run.py`(E17 Phase-0 严格协议)、`shortcut_audit.py`(E18/E18b 捷径审计,零训练,约 40 分钟)、`crpo_run.py` + `crpo_gate.py`(E19 耦合实验与闸门判定)、`phase1_disagreement.py`(待跑)
>
> 结果:全部以 JSON 保存于 `repro/`,文件名与实验对应(`table_*.json` = E16 旧协议;`rev_*.json` = E17 严格协议;`shortcut_audit.json` = E18;`crpo_*.json` = E19)。论文格式表格:`tables_paper_format.txt` 与 `table1_auroc.tex`
>
> E18/E19 复现:`python shortcut_audit.py`;`for c in ec50_assay ic50_assay; do for v in minimol minimol_mv; do python crpo_run.py --cell $c --views $v; done; done`(四进程并行约 50 分钟),然后 `python crpo_gate.py`。注意 `crpo_run.py` 以 `from revision_run import ...` 复用 E17 的数据/head/指标代码路径,不复制,以保证协议不漂移
>
> GPU 环境(Uni-Mol 特征预计算):conda env `umgpu`(torch 2.8.0+cu128,适配 RTX 5090)+ unicore + unimol;CPU 编码约 820 ms/分子,GPU + 多核并行 3D 构象后约 5–10 分钟/cell
>
> 注意:运行时必须传 `--cache_root ./cache`(源码默认写死 `/home/ubuntu/projects`);单个进程需设 `OMP_NUM_THREADS=1` 以免线程耗尽;Uni-Mol 特征必须用修复 `[MASK]` 后的 `model.py` 重新计算

---

# 第二阶段:从 RPO 到 MolRoute(2026-07-30 ~ 08-06)

*本节记录 E20 之后的全部工作。结论:原稿的中心命题失效且不可救,但项目演化出一个效应量大一个数量级的新方法。所有实验按 git tag 可追溯。*

## 9. 摘要:五句话结论

- **"pairwise 优于 pointwise"被第五次独立证伪。**继损失形式(E5)、配对结构(E7)、耦合测度(E19)之后,又在**区域限制**(E23 pAUC)与**机制鲁棒聚合**(E24 MRPO)两个维度上失败。RPO 在最终 detector bank 里被选中 8.0%、删除损失 +0.0010,与 balanced BCE(9.8% / +0.0007)无法区分。
- **Uni-Mol 的全部预训练权重只值 +0.009 macro AUROC。**完全随机初始化的 15 层 transformer 在 IC50-Assay 上与预训练版打平(Δ=−0.0003,CI 跨 0)。这是 ImageNet-OOD"未训练模型也能拾取协变量偏移"在分子上的定量复现。
- **10 个 RDKit 描述符打赢整个 512 维 MiniMol**(EC50-Scaffold 0.9773 vs 0.9699;IC50-Scaffold 0.9897 vs 0.9835),而学到的分数有效维度只有 1.3–4.2(共 512)。
- **真正的杠杆是"哪个 detector",不是"哪种损失"。**距离型与 exposure 型检测器逐机制近乎正交(Spearman −0.007~+0.126),oracle headroom 0.15–0.25。
- **MolRoute 在冻结协议下的官方 `ood_test` 上通过**:2 个 backbone × 3 个 assay × 3 档 prevalence 全部为正,MolRoute-T **+0.0611~+0.1545**,MolRoute-CF **+0.0345~+0.1530**。

## 10. 被关闭的路线(全部预注册闸门)

### E21 — Uni-Mol 编码器信息消融 `[tag: 见 §14]`

**纠正一个常见误读:**E15 的 "0.8103 vs 0.8080" 是**修复后复现值 vs 论文发表值**,是复现一致性差值,**不是消融**。修复前特征已被覆盖,该对照从未存在。本实验补上。

Phase-0 协议,4 个 primary cell,5 seed 配对,pretrained 参照取自 E17 并精确对上(macro 0.8124)。

| 臂 | macro AUROC | Δ |
|---|---:|---:|
| pretrained(193/193) | 0.8124 | — |
| `gbf_id`(3D 距离通路随机) | 0.8069 | −0.0055 |
| `emb_rand`(原子种类嵌入随机) | 0.8050 | −0.0074 |
| **`rand`(整个编码器随机)** | **0.8034** | **−0.0090** |
| `pre`([MASK] bug 的真实状态) | 0.7929 | −0.0195 |

> **结论:**Uni-Mol 全部预训练值 +0.009。`rand` > `pre`,说明该 bug 制造的是表征内部不一致,比无预训练更糟。另:E15 说 gbf "保持随机初始化"是**正确的**——`GaussianLayer.__init__` 设 mul=1/bias=0,但 `UniMolModel.__init__` 随后调用 `self.apply(init_bert_params)`(unimol.py:177)把每个 `nn.Embedding` 重设为 N(0,0.02),实测 mean −0.0005 / std 0.0198。

### E22 — 捷径子空间消融与分数有效维度

| cell | full | resid_lin | resid_poly2 | **resid_rand**(对照) | resid_pca(对照) | **desc_only** |
|---|---:|---:|---:|---:|---:|---:|
| EC50-Assay | 0.7249 | 0.7057 | 0.6928 | 0.7250 | 0.7082 | 0.6419 |
| EC50-Scaffold | 0.9699 | 0.9486 | 0.9303 | 0.9696 | 0.9533 | **0.9773** |
| IC50-Scaffold | 0.9835 | 0.9359 | 0.9134 | 0.9836 | 0.9605 | **0.9897** |
| ZINC-Scaffold | 0.5415 | 0.5778 | 0.6220 | 0.5447 | 0.6305 | 0.4352 |

随机 10 维子空间移除完全平坦(Δ≤0.001,六个 cell 全部),证伪对照通过。**预测的"scaffold 大跌 / assay 不动"分离没有出现**;真正该报的是 `desc_only`:**10 个描述符打赢 512 维 FM**。

分数有效维度(探针在 train 上拟合、test 上评估):

| cell | MLP | 线性探针 | top-1 梯度方向 | **PR**(共 512) | R²(score \| 10 desc) |
|---|---:|---:|---:|---:|---:|
| EC50-Scaffold | 0.9699 | 0.9573 | 0.9685 | **1.5** | 0.572 |
| EC50-Assay | 0.7249 | 0.6621 | 0.6378 | 4.2 | 0.064 |
| ZINC-Scaffold | 0.5415 | 0.5337 | 0.5139 | 1.6 | −0.163 |

> **方法学教训:**第一版探针在 test 上拟合又在 test 上评估,给出虚高的 0.896/0.973。**探针必须 train 拟合 / test 评估。**

### E23 — 等预算 OE 竞技场(pAUC)`[tag: 见 §14]`

6 臂,FPR95 为主指标,**等 9 试调参预算**(3 参数臂用预注册的 3 extras × 3 对角点)。`bce` 精确复现 E17。

pAUC 在多视图 scaffold 上显著优于 balanced BCE(EC50 −26%、IC50 −48% 相对 FPR95),但**逐点加权对照 `hard_bce` 拿到同样收益**(−32.5% / −37.6%),`pauc − hard_bce` 仅 2/8 显著。机制是"难例集中",不是 pairwise 结构。另:`energy_oe` 在等预算调参后不再赢 RPO(macro FPR95 0.4711 vs 0.4662)——E3 那个"2020 年方法打赢 RPO"是共享默认超参的假象,**该攻击面关闭**。

### E24 — MRPO:机制鲁棒偏好优化,闸门 6/6 FAIL `[tag: mrpo-no-go]`

作者设计的 2×2:{pointwise, pairwise} × {pooled, mechanism-robust}。robust 臂最小化 CVaR_α 的参照归一化逐机制超额风险 + KL 信赖域。

**需要重建数据:**标准 2000 分子 OE 缓存在 ~1200 个 domain 上只有 1.7 分子/assay,CVaR 会退化成难例挖掘。重建为 250/75/125 机制 × 8 分子,domain 重叠 **0**。

| | bce | rpo | cvar_bce | mrpo |
|---|---:|---:|---:|---:|
| EC50-Assay AUROC | 0.6833 | 0.6838 | 0.6842 | 0.6840 |
| IC50-Assay AUROC | 0.6549 | 0.6567 | 0.6512 | 0.6509 |

**四臂最大差距 0.0058。**因子分解:robustness 主效应 ±0.004,pairwise 主效应 ±0.002,无交互。作者预注册的"四者持平 ⇒ binary OE 路线彻底停止"触发。

### E25 — OE acquisition 两级闸门,均 FAIL `[tag: mepoe-gate0, lineB-gates]`

**Gate 0(信号是否存在):**qualified pass——headroom 4/4 通过、seed 稳定性 4/4 通过、selection value 3/4 通过、全配对 preference 迁移 4/4 不通过(0.596–0.631 vs 门槛 0.65;margin 前 50% 配对为 0.658–0.709)。subset 间方差 / seed 间方差 = 5–11×。

**Gate 1(跨 endpoint preference policy):**两个方向都 FAIL。

| target H2 AUROC | EC50→IC50 | IC50→EC50 |
|---|---:|---:|
| closest / farthest(简单启发式) | **0.5456** | **0.6047** |
| reward regression | 0.5300 | 0.5696 |
| **preference (BT/DPO)** | 0.5425 | 0.5998 |
| 零 OOD Mahalanobis | 0.5032 | **0.6562** |
| vs 最强 selector(需 ≥+0.010) | **−0.0031** | **−0.0049** |

preference 确实赢过同特征的 reward regression(+0.0126 / +0.0302),但赢不了一行代码的启发式;EC50 上免费 Mahalanobis 碾压所有 OE selector。

**效应量对比(决定了后续方向):**改变暴露哪些 OOD → 0.05~0.26;改变目标函数 → 0.001~0.004。**两个数量级。**

## 11. 转折:零 OOD 诊断

### E26 — 检测器互补性 `[tag: 无独立 tag,见 repro/zeroood_*.json]`

- **(D-A) 失效是逐机制的,不是逐 endpoint 的。**逐机制 Mahalanobis AUROC sd 0.23–0.26,EC50 有 25%、IC50 有 48% 的机制**低于随机**。
- **(D-B) target-free 预测只在同质抽样下成立。**随机切分 mean|误差| 0.018–0.023;embedding 聚类切分 0.081–0.096(q90 0.14–0.21)。⚠️ 该分析里的 Pearson r ≈ −1 是**互补切分的机械假象**,只有 mean|误差| 有意义。
- **(D-C) 距离型与 exposure 型近乎正交**:Spearman +0.005 / +0.039,条件差距 OE−Maha = +0.195/+0.315(Maha 弱处)、−0.157/−0.162(Maha 强处)。**oracle 逐机制路由上界比最好单一方法高 +0.107/+0.114。**

朴素 z-sum 融合在 EC50 上 +0.046、IC50 上 −0.019。逐分子 stacking router(Gate 0')在两个 endpoint 上无法同时为正 ⇒ 收益是**逐组切换**,不是全局组合。

## 12. MolRoute

### 12.1 方法

对 detector *d*,在独立 ID 集上标定经验 mid-CDF `F_0d`;对无标签批 *B*:

```
q_d = mean_{x∈B} F_0d(s_d(x)),     E[q_d] = (1−π)/2 + π·AUROC_d
```

批内所有 detector 共享同一 π ⇒ `argmax_d q_d` 按真实 AUROC 排序。**无需标签、无需重训、无需知道 π。**

实测(Ki,9 detector):恒等式形状成立,存在约 **0.039 的常数偏移**(ID_cal 与 ID_test 分布差异),但九个 detector 的偏移彼此只差约 0.003,**在 argmax 中抵消**。

两个推理协议:
- **MolRoute-T**(transductive):整批选择、整批打分。ranking 最强,但选择与打分耦合。
- **MolRoute-CF**(cross-fitted):A 半选择给 B 半打分、B 半选择给 A 半打分,在共同 ID-ECDF 尺度合并。**大幅降低选择导致的假阳性膨胀**(不是校准保证)。

### 12.2 E27 — 学习式 router 不是必要的 `[tag: decisive-ecdf]`

⚠️ **必须避免的退化设定:**若每个 batch 恰为一个完整机制(100% OOD),则各 detector 的 AUROC 本身就是可直接计算的两样本统计量,oracle 在测试时白送,"选择"是平凡的。该版本给出 +0.087/+0.125、selector 准确率 0.89,是**循环论证的假象**。必须用**成分未知的混合批**、目标为**批内 AUROC**。

clean splits(ID_fit 1200 / ID_cal 800 / ID_router 500 / ID_final 500,两两不相交),5 cell × 5 seed:

| | ECDF(解析) | GBDR(学习) | GBDR−ECDF |
|---|---:|---:|---:|
| 三个 assay 50% | +0.125 / +0.149 / +0.154 | — | 均值 **+0.0070** |
| 三个 assay 25% | +0.107 / +0.119 / +0.118 | — | 均值 **−0.0015** |
| 三个 assay 10% | +0.035 / +0.086 / +0.072 | — | 均值 **−0.0014** |

符号跨 cell 不一致 ⇒ 预注册的 "ECDF ≈ GBDR" 情形,**放弃 learned-router 叙事**。

**同时修正的三个协议缺陷**(旧 `batchsel_multi.py` 结果标为 exploratory):历史批与测试批共用同一 `id_test` 池;`best_fixed` 是对**测试**矩阵取 max(是 oracle 不是可部署基线,已改名 `hist_best`);ECDF 参照与 detector 拟合集同源。

### 12.3 E28 — bank 消融:RPO 不获得特殊地位 `[tag: bank-ablation]`

pairwise-logistic RPO **原本不在 bank 里**,补入后(同 head/优化器/预算,唯一差别是损失):

| detector | 被选中 | 删除后损失 |
|---|---:|---:|
| **OE-MV** | 7.5% | **+0.0098** |
| **Mahalanobis** | 21.0% | **+0.0079** |
| LOF | 12.7% | +0.0028 |
| OE-pAUC | 14.7% | +0.0016 |
| **RPO** | **8.0%** | **+0.0010** |
| **OE (balanced BCE)** | **9.8%** | **+0.0007** |
| Energy / MSP / ODIN / KNN | 3–9% | ≈0 |

⚠️ 单个 cell/seed 曾显示 RPO 11.2% vs BCE 6.6%,**全 sweep 下不成立,不要引用**。

### 12.4 E29 — 官方 `ood_test`,协议先冻结 `[tags: molroute-protocol-frozen → molroute-oodtest-pass]`

`PROTOCOL_MOLROUTE.md` 在**任何测试分子被打分之前**写定并打 tag:路由 bank 固定 {OE-MV, Mahalanobis, LOF, OE-pAUC};开发仅用 `ood_val`;`historical_best` 与 guard 各为**每 cell 一个决定**、在混合 prevalence 上定;测试用 400 机制/cell、seed 777。

审计:五个 cell 的 domain 重叠与 canonical-SMILES 重叠(vs 开发 OOD / ID_train / ID_test)**全部为 0**。

### 12.5 E30 — sanity `[tag: molroute-sanity]`

**纯 ID 批(π=0):**选择确实膨胀假阳性,FPR@5% 增量 +0.009~+0.036;半分选择基本消除。

**多机制批(OOD 总数固定 8):**

| | oracle headroom | 捕获 | **捕获比例** |
|---|---:|---:|---:|
| 1 机制 ×8 | 0.2544 | +0.1237 | **48.4%** |
| 2 机制 ×4 | 0.1957 | +0.0622 | **31.4%** |
| 4 机制 ×2 | **0.1538** | +0.0114 | **7.2%** |

⚠️ **headroom 并未消失**(仅压缩 40%),塌掉的是捕获比例(85%)。正确表述:异质性压缩 detector 间距,同时让固定批大小下的无标签路由信噪比崩溃——**不是可选择空间消失**。

### 12.6 E31 — CF 与校准分解 `[tag: molroute-cf]`

CF 保留 86% 的 transductive 收益,50% prevalence 下 97–100%。

校准漂移与选择膨胀分离(ID_cal 内部半分作同池参照):

| cell | 名义 | 漂移 | **选择膨胀** |
|---|---|---:|---:|
| EC50 | 5% | +0.0295 | +0.0202 |
| IC50 | 5% | +0.0075 | +0.0184 |
| Ki | 5% | **−0.0165** | +0.0217 |

Ki 的低 FPR 主要是**校准漂移**,不是 routing。选择膨胀在三个 assay cell 上稳定 **1.8–2.2 个百分点**(5%)。

### 12.7 E32 — Uni-Mol 复现 `[tag: molroute-unimol]`

Uni-Mol 对测试分子覆盖率 **100.00%**(五个 cell),跨 backbone 交集即完整 3200 分子/cell。`historical_best` 与 guard 由各自 backbone 的 `ood_val` 独立重新生成。

| | MiniMol −T / −CF | Uni-Mol −T / −CF |
|---|---|---|
| assay 25%/50% 均值 | **+0.1260 / +0.1162** | **+0.1187 / +0.1096** |
| assay 10% 均值 | +0.0741 / +0.0560 | +0.0830 / +0.0569 |
| scaffold(guard FALLBACK) | 0.0000 | 0.0000 |

**额外证据:`historical_best` 的身份在 5 个 cell 里有 4 个跨 backbone 就变**(MiniMol: Mahalanobis / OE / RPO / OE-pAUC / RPO;Uni-Mol: OE-MV / LOF / OE-pAUC / OE / RPO)。guard 在两个 backbone 上对五个 cell 做出完全相同的 ROUTE/FALLBACK 判断。

### 12.8 E33 — canonical export 与论文数据冻结 `[tag: molroute-paper-data-v1]`

修四点:补齐逐机制日志(原 `per_mech` 被初始化但从未写入);统一 canonical T(用与 CF matched 的版本,旧 `molroute.py` 导出因 ID 重采样差最多 0.008,**已标 superseded**);oracle 拆为 `oracle_routing`(4 个路由成员,**可达上限**)与 `oracle_full`(全部 10 个,仅作背景);CF 措辞改为"substantially reduces selection-induced false-positive inflation"。

新增两个基线:**mean-ensemble 不能替代选择**——assay 上 0.577–0.635(≤HistBest),scaffold 上损失 8–15 个点;**GBDR 在官方测试上与 T 互有胜负**(6 个配置各赢 3),即解析选择器**匹配**学习式 router 而非击败它。

按**可达** oracle 算的捕获率:MiniMol T 37.9/57.1/64.0%,Uni-Mol T 45.2/58.0/62.2%(10/25/50%)。

## 13. 论文口径(冻结)

**中心 claim:**

> MolRoute shows that detector quality is conditional on the molecular mechanism and that this conditionality can be exploited using unlabeled target batches, yielding +0.06–0.14 AUROC over a detector selected on development mechanisms.

**摘要数字:**MolRoute-T **+0.0611~+0.1545**;MolRoute-CF **+0.0345~+0.1530**(prevalence ≥25% 时 +0.0706~+0.1530);CF 均值 +0.0565 / +0.0947 / +0.1311(10/25/50%)。

**RPO 的位置(不再讨论 conditional complementarity):**

> RPO remains a competitive fixed detector—it is the historical best on Ki-Assay and IC50-Scaffold—but no fixed detector is uniformly reliable. MolRoute further improves over RPO by +0.143 AUROC on Ki-Assay.

**必须限定:**"无标签"指不使用 target-batch 标签(bank 与 guard 仍用历史 OE 数据);10% prevalence 已实测有效;双机制仍有 +0.062;四机制收益基本消失;显著改善但仍留 0.15–0.25 的 oracle headroom。

**Novelty 定位:**不可写"无标签 detector selection 无先例"。已有 MetaOD (NeurIPS'21)、MetaOOD (ICLR'25)、Gscore。可守的是 *target-label-free, batch-adaptive detector selection under molecular mechanism shift*,靠小 batch、同 endpoint 内逐机制路由、2D/3D/OE 异构 bank、有限样本安全回退区分。

**原稿必须作废的 claim:**RPO 普遍优于 matched BCE;preference optimization 是主要性能来源;原表提升证明 pairwise 有效;广义 scientific FM reliability / unseen-shift robustness。

⚠️ **Figure 1 必须删除。**`run_ablation_study.py:98` 的注释原文为 `# BCE performance degradation`,hinge 与 BCE 的 lr 差 40 倍、正则差 50000 倍,且用的是 squared top-50% hinge 而非 RPO。

⚠️ **PDF 隐藏文字非作者所为。**注入句与 "Confidential reviewer copy" 页脚的 y 坐标**六位小数完全相同**(754.539795 / 761.574707),字体为 ToUnicode 重映射子集;正文页为干净 Times。它在 OpenReview 加盖图层里,**重新编译无法去除,也无需去除**。

⚠️ **Ki 不再是密封 endpoint。**已在 batchsel 中开启并影响了 guard 与 bank 设计,应称 "additional evaluation endpoint"。目前无任何未开封 endpoint(potency 仅 9 个 domain;GOOD 仅 56–119 个 scaffold 组),下一稿的密封测试需新建(ChEMBL 时间切分或外部数据集)。

## 14. 待办

- **投稿前唯一必补:**无标签 detector selection 一支的对照——MetaOD / MetaOOD / Gscore。需先确认其确切定义,否则只能做标注清楚的近似复现。
- 双-backbone(14 detector)bank:3 seed 下 Ki +0.180,但会改变 bank 大小、复杂化故事,列为扩展。
- `MANIFEST_molroute_paper_data_v1.json` 的 `commit` 字段指向父提交 `dd49194`(哈希在提交前计算),溯源正确但非自洽,可重算。
- 论文形态:**建议按新论文写和投,LaTeX 工程复用作骨架。**可逐字搬:实验设置、检测器实现、数据划分、复现说明、RPO 实现(移入附录)。需新写:Intro、Method、Fig 1/2、Table 1、Limitations、Related Work 新增一节。

## 15. 第二阶段脚本与产物

| 脚本 | 作用 | 产物 |
|---|---|---|
| `ablate_unimol.py` + `diag_encoder.py` | E21 编码器消融 | `repro/encabl_*.json` |
| `diag_subspace.py` | E22 捷径子空间 + 分数维度 | `repro/diag_*_fm.json` |
| `arena.py` | E23 等预算 OE 竞技场 | `repro/arena_*.json` |
| `mrpo_data.py` + `mrpo.py` | E24 MRPO 2×2 | `repro/mrpo_*.json` |
| `mepoe_data.py` + `mepoe_gate0.py` | E25 Gate 0 | `repro/mepoe_gate0_*.json` |
| `routing_gate.py` + `mepoe_gate1.py` | E25 Gate 0' / Gate 1 | `repro/routing_*.json`, `repro/gate1_*.json` |
| `zeroood_diag.py` | E26 零 OOD 诊断 | `repro/zeroood_*.json` |
| `batchsel.py` / `batchsel_multi.py` | E27 批级选择(exploratory) | `repro/batchmulti_*.json` |
| `decisive.py` | E27 ECDF vs GBDR 决定性对照 | `repro/decisive_*.json` |
| `bankablate.py` | E28 bank 消融 | `repro/bankablate_*.json` |
| `molroute_data.py` + `molroute.py` | E29 官方 ood_test | `repro/molroute_*.json` |
| `sanity.py` | E30 纯 ID / 多机制 | `repro/sanity_*.json` |
| `molroute_cf.py` | E31 CF + 校准分解 | `repro/molroutecf_*.json` |
| `molroute_unimol_enc.py` | E32 Uni-Mol 编码 | 特征缓存 |
| `molroute_export.py` | E33 canonical export | `repro/export_*.json` |
| `paperdata.py` | 论文数据聚合 | `repro/paperdata.json`, `MANIFEST_*.json` |

> 分支 `revision/oe-acquisition`。tag 顺序:`mrpo-no-go` → `mepoe-gate0` → `lineB-gates` → `decisive-ecdf` → `bank-ablation` → **`molroute-protocol-frozen`** → `molroute-oodtest-pass` → `molroute-sanity` → `molroute-cf` → `molroute-unimol` → **`molroute-paper-data-v1`**
>
> 图表(交互式,含 hover 与表格视图):https://claude.ai/code/artifact/35073bf5-9f2b-46a1-b06b-6c324a4cafab
>
> 环境补充:配色校验器需 node,装在**独立** conda env `nodeenv`(装进 base 会破坏 conda 的 libmamba solver)。MiniMol 编码约 182 ms/分子且内部多线程,并发进程数需控制。后台任务必须 `setsid` 启动,否则监控命令超时的 SIGTERM 会连带杀死子进程。

---

# 第三阶段:matched-OE 主表与"对 RPO 有利的 setting"(2026-08-06)

*目标:按作者要求做最小修改原稿的路线——把 BCE 补成 matched baseline、把原稿全部 baseline 做 OE 化,并主动构造理论上对 pairwise 有利的 setting。全部预注册,脚本 `matched_oe.py` / `capacity_sweep.py` / `favorable.py`,结果 `repro/moe_*.json` / `cap_*.json` / `fav_*.json`,聚合 `agg_matched_oe.py` / `agg_favorable.py`。*

## 16. 摘要:四句话结论

- **matched-OE 主表做成了,而且它自己否定了原稿的中心主张。**OE 化是主效应(Mahalanobis +0.266、Energy +0.224、MSP +0.195 macro AUROC);RPO 与 Balanced OOD Head 在 12 cell × 2 backbone 上 **24/24 的 95% CI 全部跨 0**;两个干净 assay cell 上 RPO 在两个 backbone 上**都排第 4**,输给零训练的 `OE-Mahalanobis` / `OE-KNN`。
- **head 容量是全项目第一个跨 cell 一致的正向信号,但幅度只有 +0.006。**h=4(2,077 参数)时 RPO−BOH 在 14 个 cell×backbone 里 11 个为正(binom p=0.057;剔除已知方向翻转的 ZINC-Scaf/MiniMol 后 11/13,p=0.022,均值 +0.0060);full head 时均值 −0.0011、7/14 为正。
- **预注册的形状检验全部不通过。**Axis A(Δ 随容量单调衰减)20 个检验里 12 个方向正确但只有 1 个显著、**3 个显著反向**(binom p=0.503);Axis B(Δ 随 OE 难度单调上升)24 个检验 14 个方向正确、1 个显著、binom p=0.541。
- **tail-oriented selection 的正向信号是选择噪声,不是 pairwise 结构。**120 次比较里 7 次达到 Δ≥+0.010 且 5/5 seed(≈噪声期望),命中位置在 h/tier/sel 上完全散乱;最大的一格 (EC50/h=4/far/tail) 上 pointwise 的 `hard_bce` 拿到 **+0.0417 > RPO 的 +0.0383**。

## 17. E34 — matched-OE 主表 `[matched_oe.py]`

14 个方法,同冻结特征、同 ID/OOD 子集、同 domain-disjoint 验证切分、**每个方法恰好 9 次验证试验**、3 选择 seed + 5 配对最终 seed。原稿六个 baseline 各配一个 OE 版本:

| 原 baseline | OE 版本 | 构造 |
|---|---|---|
| MSP | MSP-OE | 辅助 OOD 上加均匀输出损失(Hendrycks OE) |
| ODIN | ODIN-OE | 在同一个 OE-trained classifier 上做温度缩放 |
| Energy | Energy-OE | ID CE + Liu 2020 能量边界 |
| Mahalanobis | OE-Mahalanobis | 两样本高斯对数似然比(shared / QDA / diagonal 三种协方差估计) |
| KNN | OE-KNN | 到 ID 的 kNN 距离 − 到辅助 OOD 的 kNN 距离 |
| LOF | OE-LOF | ID 与 OOD 两个局部密度模型的 outlierness 之差 |

⚠️ 后三者**不是文献中的标准方法**,论文里必须写成 *two-sample OE adaptations*,不可冒充官方方法。

**GOOD 的 cls_label 缺失问题已解决**(`good_labels.py`):从 `data/GOOD*/{scaffold,size}/processed/*.pt` 提取 SMILES→标签。HIV 用原二值活性;PCBA 取**少数类最大**的 assay 列(task 93,239375/62800;注意"最密列"是 411796/71,会训出退化分类器);ZINC 是连续目标,按 ID 中位数 −2.1281 二值化,**属于本文的适配,必须在论文里说明**。

**主表(MiniMol,7 个非 size cell macro AUROC):**

| 方法 | OE | EC50-Assay | IC50-Assay | EC50-Scaf | IC50-Scaf | HIV-Scaf | PCBA-Scaf | ZINC-Scaf | macro |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSP | × | 0.4899 | 0.5695 | 0.6459 | 0.6556 | 0.5300 | 0.4488 | 0.4402 | 0.5400 |
| ODIN | × | 0.4618 | 0.5876 | 0.6276 | 0.6624 | 0.5047 | 0.4585 | 0.4402 | 0.5347 |
| Energy | × | 0.4760 | 0.5692 | 0.6460 | 0.6557 | 0.5140 | 0.4488 | 0.4459 | 0.5365 |
| Mahalanobis | × | 0.6084 | 0.5177 | 0.5851 | 0.5365 | 0.3575 | 0.5065 | 0.4858 | 0.5139 |
| KNN | × | 0.6536 | 0.5802 | 0.7120 | 0.6597 | 0.4714 | 0.5905 | 0.5074 | 0.5964 |
| LOF | × | 0.5924 | 0.5426 | 0.6604 | 0.6857 | 0.5057 | 0.5634 | 0.5859 | 0.5909 |
| MSP-OE | ✓ | 0.6094 | 0.5983 | 0.9435 | 0.9513 | 0.7297 | 0.8127 | 0.4982 | 0.7347 |
| ODIN-OE | ✓ | 0.6094 | 0.5983 | 0.9351 | 0.9422 | 0.7304 | 0.8183 | 0.4976 | 0.7330 |
| Energy-OE | ✓ | 0.6676 | 0.6034 | 0.9554 | 0.9590 | 0.7481 | 0.8400 | 0.5463 | 0.7600 |
| **OE-Mahalanobis** | ✓ | **0.7660** | **0.6805** | 0.9153 | 0.9831 | 0.7611 | 0.8994 | 0.4560 | 0.7802 |
| OE-KNN | ✓ | 0.7460 | 0.6515 | 0.9037 | 0.9339 | 0.7084 | 0.8479 | 0.4787 | 0.7529 |
| OE-LOF | ✓ | 0.6385 | 0.5771 | 0.9051 | 0.9238 | 0.7006 | 0.7909 | 0.4616 | 0.7139 |
| Balanced OOD Head | ✓ | 0.7249 | 0.6584 | **0.9699** | 0.9835 | **0.7625** | 0.9018 | 0.5415 | **0.7918** |
| RPO | ✓ | 0.7264 | 0.6564 | 0.9693 | **0.9841** | 0.7581 | **0.9019** | **0.5430** | 0.7913 |

**两个干净 assay cell 的排名(macro AUROC):**

| # | MiniMol | | # | Uni-Mol | |
|---:|---|---:|---:|---|---:|
| 1 | OE-Mahalanobis | 0.7232 | 1 | OE-KNN | 0.6483 |
| 2 | OE-KNN | 0.6987 | 2 | OE-Mahalanobis | 0.6458 |
| 3 | Balanced OOD Head | 0.6917 | 3 | Balanced OOD Head | 0.6409 |
| **4** | **RPO** | **0.6914** | **4** | **RPO** | **0.6404** |

FPR95 同向(MiniMol assay macro:OE-Maha 0.8213 < RPO 0.8389 < BOH 0.8457)。

**RPO − Balanced OOD Head 的 24 个配对检验:12 个 cell × 2 backbone,95% CI 无一不跨 0**,最大 |Δ| = 0.0045(HIV-Scaffold/MiniMol,负)。这是本项目**第六次**独立证伪。

> **结论(可直接进正文):**(1) 原表相对 post-hoc 基线的巨大提升几乎全部来自 outlier exposure,审稿人的信息不对称质疑在数据上完全成立;(2) 一旦补齐 OE,两个**零训练**的 two-sample 距离基线在最干净的 cell 上反超 RPO 达 **+0.0396**;(3) 最优方法的身份跨 cell 与 backbone 不稳定(OE-Maha 在 EC50-Assay/MiniMol 领先 4 个百分点,在 ZINC-Scaffold 掉到 0.4560 < BOH 0.5415),与 §12.7 的 `historical_best` 跨 backbone 变身是同一现象。

**副产品:**Uni-Mol 上 ID-only 的 LOF 拿到 size macro 0.9610、scaffold 也很高,而 MiniMol 上只有 0.8246 —— size/scaffold 的捷径在 3D 表征里更暴露,与 E18 捷径审计互相印证。

## 18. E35 — RPO-Lite 容量扫 `[capacity_sweep.py]`

`RR.Head` 被 monkey-patch 成宽度 h 的轻量 head(投影总宽与打分宽都是 h),`RR.train`/`RR.run_objective` 逐字节保持 Phase-0 协议,协议漂移不可能发生。7 个非 size cell × 2 backbone × 8 个宽度 × 2 目标,等 9 试预算,5 配对 seed。

**Δ = RPO − Balanced OOD Head:**

| cell | | h=4 | h=8 | h=16 | h=32 | h=64 | h=128 | h=256 | full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EC50-Assay | mini | +0.0048 | +0.0024 | +0.0014 | −0.0053 | −0.0007 | +0.0012 | +0.0001 | +0.0015 |
| EC50-Assay | uni | +0.0153 | +0.0019 | −0.0012 | +0.0047 | +0.0031 | +0.0030 | −0.0019 | −0.0054 |
| IC50-Assay | mini | +0.0148 | −0.0074 | −0.0009 | +0.0026 | −0.0014 | −0.0018 | −0.0007 | −0.0019 |
| IC50-Assay | uni | +0.0060 | −0.0033 | +0.0000 | +0.0019 | +0.0035 | +0.0004 | +0.0044 | +0.0044 |
| EC50-Scaf | mini | +0.0013 | +0.0075 | −0.0053 | −0.0058 | −0.0011 | +0.0002 | +0.0004 | −0.0006 |
| EC50-Scaf | uni | +0.0006 | −0.0028 | +0.0144 | −0.0064 | −0.0014 | −0.0032 | −0.0009 | −0.0054 |
| IC50-Scaf | mini | −0.0003 | +0.0013 | −0.0005 | +0.0006 | +0.0008 | +0.0005 | −0.0002 | +0.0006 |
| IC50-Scaf | uni | +0.0095 | +0.0003 | −0.0014 | −0.0010 | −0.0016 | −0.0017 | −0.0020 | −0.0022 |
| HIV-Scaf | mini | +0.0164 | −0.0005 | +0.0014 | −0.0008 | −0.0024 | −0.0012 | +0.0013 | −0.0045 |
| HIV-Scaf | uni | −0.0124 | −0.0065 | −0.0077 | −0.0013 | −0.0054 | −0.0038 | +0.0047 | +0.0010 |
| PCBA-Scaf | mini | +0.0103 | −0.0010 | −0.0043 | −0.0032 | −0.0019 | −0.0003 | −0.0010 | +0.0001 |
| PCBA-Scaf | uni | +0.0053 | −0.0032 | −0.0057 | −0.0043 | −0.0090 | −0.0059 | −0.0005 | +0.0042 |
| ZINC-Scaf | mini | **−0.0415** | −0.0167 | +0.0053 | +0.0038 | +0.0121 | +0.0148 | +0.0017 | +0.0015 |
| ZINC-Scaf | uni | +0.0064 | +0.0023 | +0.0050 | −0.0008 | −0.0064 | −0.0006 | −0.0045 | −0.0085 |

**h=4(2,077 参数)是唯一有一致信号的一档:11/14 为正,均值 +0.0026(binom p=0.057,Wilcoxon p=0.091);剔除 ZINC-Scaf/MiniMol 后 11/13、均值 +0.0060、p=0.022。full head 一列均值 −0.0011、7/14 为正。**

**但预注册的形状检验不通过:**Spearman(width, Δ) 在 14 个配置里 9 个为负(binom p=0.21,不显著),只有 4 个单独显著,而 **HIV-Scaffold/Uni-Mol 显著为正(ρ=+0.881, p=0.004)**。因此只能说"效应集中在 h=4",**不能说 Δ 随容量单调衰减**。

⚠️ 写作上必须同时给出绝对值:h=4 时 EC50-Assay 的 AUROC 是 0.6873,比 full head 的 0.7264 低 **0.039**。也就是说 pairwise 的相对优势出现在一个**没人会实际部署**的容量档上,parameter-efficiency 的说法必须连同这个代价一起报告。

## 19. E36 — 三条"对 RPO 有利"的轴 `[favorable.py]`

2 个干净 assay cell × 6 个宽度 × 5 个曝光层 × 6 个目标 × 2 个选择规则 = 120 次比较。曝光分层用**纯 ID 统计量**(到 20 个最近 ID 训练分子的平均距离)切成 near/medium/far 各 30%,外加**同预算的 random 对照**与全池 `all`;val/test OOD 完全不变,只有曝光在变。两种 FPR 均按明确命名报告:`FPR-OOD@95TPR-ID` 与 `FPR-ID@95TPR-OOD`。

**必须带的 pointwise 杀手对照:**`hard_bce`(top-50% 难例)与 `focal_bce`。E23 已证明"集中在难例"是 pointwise losses 完全能表达的,若它们追平 RPO,则无论 Δ 多大都不构成 pairwise 结构的证据。等预算:所有目标共享同一个 3×3 (γ×lr) 网格 = 9 试,`hard_bce` 的 q、`focal_bce` 的 γ_f、`pauc` 的 α、`energy_margin` 的边界全部**固定在先验值,不参与调参**。

| 轴 | 预注册预测 | 结果 | 判定 |
|---|---|---|---|
| A 容量 | Spearman(width, Δ) < 0,两个 cell 同号 | 20 个检验中 12 个方向正确,**1 个显著正确、3 个显著反向**,binom p=0.503 | **FAIL** |
| B OE 难度 | Spearman(难度, Δ) > 0,两个 cell 同号 | 24 个检验中 14 个方向正确,1 个显著正确,binom p=0.541 | **FAIL** |
| C tail 选择 | Δ≥+0.010 且 5/5 seed 的格子应集中出现 | 120 次里 **7 次**命中(≈噪声期望),位置在 h/tier/sel 上完全散乱 | **FAIL** |

**Axis C 的对照列是决定性的:**

| cell | width | tier | sel | Δ RPO | Δ hard_bce | Δ focal_bce | Δ pAUC | pairwise 专属? |
|---|---|---|---|---:|---:|---:|---:|:--:|
| EC50 | h=4 | far | tail | +0.0383 | **+0.0417** | +0.0257 | +0.0209 | 否 |
| EC50 | h=4 | all | tail | +0.0162 | −0.0465 | −0.0207 | −0.1085 | 是 |
| IC50 | h=16 | medium | tail | +0.0177 | −0.0344 | +0.0078 | −0.0178 | 是 |
| IC50 | h=128 | far | auroc | +0.0156 | +0.0093 | −0.0054 | +0.0093 | 是 |
| IC50 | h=128 | far | tail | +0.0111 | **+0.0123** | −0.0018 | +0.0047 | 否 |
| IC50 | h=128 | medium | auroc | +0.0111 | −0.0037 | −0.0129 | −0.0153 | 是 |
| IC50 | full | far | tail | +0.0255 | +0.0179 | +0.0229 | +0.0239 | 是 |

**最大的一格上 pointwise 的 `hard_bce` 反超 RPO**;5 个"pairwise 专属"的格子分布在 h=4/16/128/full 与 far/medium/all,没有任何可解释的结构 —— 这是散点噪声的形态,不是机制的形态。

⚠️ **必须报告的混淆:**`random` 层的 BCE 绝对 AUROC(EC50 0.6684 / IC50 0.6494)显著高于 near/medium/far(0.57–0.61)。near/medium/far 是距离上的**连续切片**,在改变难度的同时也降低了曝光集的**多样性**。因此 Axis B 的任何效应都同时混着"难度"和"多样性",同预算的 `random` 对照把这一点暴露了出来,论文中不得只报告 near-OOD。

**关于 `arena` 的 +0.0178/+0.0129 的更正:**该数字属实(`repro/arena_{ec50,ic50}_assay_minimol_equal.json`,`sel_fpr95`),但机制不是 RPO 变好:RPO 在两种选择规则下选中**同一组超参、数值完全相同**(EC50 0.7264),变化的是 **BCE 被 tail 准则选到更差的配置**(0.7249→0.7087);同一 cell 上 `hard_bce` 拿到 **+0.0261 > RPO**;换到 MV 视图 RPO 只剩 +0.0052 / −0.0043。因此它是选择噪声,**不可作为正向证据引用**。

## 20. 第三阶段对论文口径的影响

- **原稿"最小修改"路线在技术上完成、在结论上失败。**matched-OE 主表做出来了,公平性攻击面彻底封住,但这张表本身证明 pairwise 目标不是性能来源,并且两个零训练基线在最干净的 cell 上超过 RPO。**该表一旦进正文,RPO 就不能作为主张被保留。**
- **`OE-Mahalanobis` / `OE-KNN` 反超是新增的、必须正面处理的问题**,它比原来的"RPO ≈ BCE"更硬:审稿人只要照做一次就能复现。
- **唯一可保留的正向表述**(且必须严格限定):在 2,077 参数的极小检测头上,pairwise 目标比 balanced BCE 平均高 +0.006 AUROC(11/13 配置一致,p=0.022),但该容量档的绝对性能比完整 head 低 0.039;**这不构成方法主张,只能作为一条观察写进 discussion**。
- **不得写入论文的表述:**"tail-oriented selection 下 RPO 优于 BCE"(选择噪声,且 pointwise 对照反超);"Δ 随容量单调衰减"(形状检验 FAIL,且有显著反向配置);"near-OOD 曝光下 pairwise 更优"(Axis B FAIL,且与多样性混淆)。

## 21. 第三阶段脚本与产物

| 脚本 | 作用 | 产物 |
|---|---|---|
| `good_labels.py` | 提取 GOOD 三个数据集的 SMILES→标签 | `cache/good_labels_{hiv,pcba,zinc}.json` |
| `matched_oe.py` | E34 matched-OE 主表(14 方法 × 12 cell × 2 backbone) | `repro/moe_*.json` |
| `capacity_sweep.py` | E35 RPO-Lite 容量扫(8 宽度 × 7 cell × 2 backbone) | `repro/cap_*.json` |
| `favorable.py` | E36 容量 × OE 难度 × 选择规则(60 配置) | `repro/fav_*.json` |
| `agg_matched_oe.py` | E34/E35 聚合与闸门判定 | `repro/matched_oe_tables.md`, `matched_oe_summary.json` |
| `agg_favorable.py` | E36 聚合与三条轴的形状检验 | `repro/favorable_tables.md`, `favorable_summary.json` |
| `run_matched_oe.sh` / `run_favorable.sh` | 并行启动器(OMP=1,setsid) | `logs/` |

> 复现:`bash run_matched_oe.sh`(38 进程,约 2 小时)、`bash run_favorable.sh`(60 任务,xargs -P 40,约 2 小时),然后 `python agg_matched_oe.py` 与 `python agg_favorable.py`。解释器 `/root/miniconda3/envs/ood/bin/python`。

## 22. E37 — Reference-Anchored RPO:把最强 matched-OE 检测器当作 DPO 的 reference `[reference_rpo.py]`

**动机.** E34 把两个零训练的 two-sample 检测器排在 RPO 前面(OE-Mahalanobis 第一)。把它当**竞争对手**必输;把它当 **reference policy**——DPO 的核心成分,恰恰是原始 RPO 缺的那一块——则只需学习 reference 排错的 ranking residual:

> s_θ(x) = m(x) + α·g_θ(h(x)),  m = 冻结的 OE-Mahalanobis(z-score);
> w_io = σ(−τ[m(x_o) − m(x_i)]),  归一化到均值 1;
> L = mean_io w_io·softplus(−β[s(x_o) − s(x_i)]) + λ·E[g²]

**温度标定(正式实验前的数值设计修正,已写入 protocol).** 固定 τ 不可用:z-score 后典型 pair 的 `m_o − m_i` 在 1–3,τ=5 给出 σ(−10)≈5e-5,梯度塌缩到极少数 pair(实测 minibatch ESS 中位数 **0.019**)。改为**只用训练侧 reference margin** 标定:取满足 `r(τ) = (Σw)²/(N·Σw²) = 0.5` 的最小 τ(二分,r 对 τ 单调递减),不使用任何验证/测试结果。
目标必须是 0.5 不是 0.25:**τ→∞ 时 w → 1[m_o < m_i],故 r 的下确界 = 1 − AUROC_ref ≈ 0.24–0.26**,0.25 压在渐近线上。此判断已被实测证实——`ic50_assay/unimol` 上 ESS=0.25 **无解**,τ 顶到二分上界,mbESS 停在 0.323 ≈ 1 − AUROC_ref。ESS∈{0.25,0.75} 与固定 τ∈{1,5,10} 仅作敏感性,不参与选择。

**六级阶梯 + 对照**(同 reference、同 head、同辅助数据、每臂**恰好 9 试** α×lr;τ 是标定不是调参):

`ref_only`(g=0)→ `ref_bce`(pointwise residual)→ `ref_weighted_bce`(pair 权重边际化成单样本权重 u_o=E_i[w_io], u_i=E_o[w_io],排除"只是 hard-example weighting")→ **`fusion`**(独立训练的原始 RPO head,训练时看不到 m,事后按验证集选线性权重相加——与 `ref_rpo` 的唯一差别是 pairwise 损失作用在总分还是只作用在 residual)→ `ref_rpo`(未加权,即 **Reference-Anchored RPO**)→ `ref_rpo_ess`(完整加权版)。

**Ki-Assay 作为冻结后的确认 endpoint.** 它此前没有 seed42 splits;按 `utils.process_drugood_data` + `data_loader._prepare_splits` 的原始构造重建(`build_ki_splits.py`),并**用同一脚本重建 ec50_assay 与既有缓存逐元素比对,六个 split 全部 identical** 后才生成 Ki。其特征缓存原为 MolRoute 机制子集所建、仅覆盖 27%(train_id 73/2000),故用 `encode_cell.py` 补编码,MiniMol 与 Uni-Mol 均达 **6/6 split 100% 覆盖、零批次失败**。按 §13 口径 Ki 只能称 additional evaluation endpoint,不称 sealed,因此**与开发四格分开报告、不混算 macro**。

**主表 — 单一固定方法的 macro AUROC**(逐格"最强外部基线"是 test oracle,不作为比较对象):

| 方法 | EC50/mini | EC50/uni | IC50/mini | IC50/uni | 开发 macro | Ki/mini | Ki/uni | Ki macro | 六格 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OE-Mahalanobis(reference) | 0.7660 | 0.6621 | 0.6805 | 0.6295 | 0.6845 | **0.7063** | **0.6563** | **0.6813** | **0.6835** |
| Reference-BCE | 0.7708 | 0.6618 | 0.6627 | 0.6314 | 0.6817 | 0.7018 | 0.6514 | 0.6766 | 0.6800 |
| Reference-weighted BCE | 0.7593 | 0.6646 | 0.6610 | 0.6234 | 0.6771 | 0.6934 | 0.6462 | 0.6698 | 0.6747 |
| Reference + independent RPO(fusion) | 0.7697 | 0.6596 | **0.6955** | **0.6368** | **0.6904** | 0.6999 | 0.6332 | 0.6665 | 0.6824 |
| **Reference-Anchored RPO** | **0.7710** | 0.6596 | 0.6878 | 0.6297 | 0.6870 | 0.7043 | 0.6464 | 0.6753 | 0.6831 |
| Reference-weighted RPO(ESS) | 0.7642 | 0.6564 | 0.6781 | 0.6237 | 0.6806 | 0.7041 | 0.6373 | 0.6707 | 0.6773 |
| OE-KNN | 0.7460 | **0.6683** | 0.6515 | 0.6283 | 0.6735 | 0.7117 | 0.6050 | 0.6583 | 0.6685 |
| Balanced OOD Head | 0.7249 | 0.6526 | 0.6584 | 0.6292 | 0.6663 | 0.6235 | 0.6110 | 0.6173 | 0.6499 |
| 原始 RPO | 0.7264 | 0.6472 | 0.6564 | 0.6336 | 0.6659 | 0.6437 | 0.6125 | 0.6281 | 0.6533 |
| Energy-OE | 0.6676 | 0.6036 | 0.6034 | 0.5551 | 0.6074 | 0.6172 | 0.5991 | 0.6082 | 0.6077 |
| OE-LOF | 0.6385 | 0.6130 | 0.5771 | 0.5550 | 0.5959 | 0.5997 | 0.5567 | 0.5782 | 0.5900 |

**四条预注册决策规则,全部不通过:**

| 规则 | 开发四格 | Ki 两格 | 判定 |
|---|---|---|---|
| 必须占据单一方法 macro 第一 | 第 2(`fusion` 0.6904 在前) | 第 4 | **FAIL** |
| 必须高于 Reference-BCE | macro +0.0053,但**仅 2/4 格为正**(全部由 IC50/mini 的 +0.0252 撑起) | macro **−0.0013**,1/2 | **FAIL** |
| 必须打赢 `fusion` 对照 | macro **−0.0034**,2/4 | +0.0088,2/2 | **FAIL(开发格)** |
| 相对 reference 为正 | +0.0025,3/4 | **−0.0060,0/2** | **开发信号在确认 endpoint 上反号** |

**最重要的一条:开发四格上 +0.0025/3-of-4 的优势,在 Ki 上翻成 −0.0060/0-of-2。**冻结后的确认 endpoint 正好抓住了一个开发集假象——这与 E17(domain 重叠导致双层验证过的结论反转)是同一类教训。

**reference 加权本身有害,且单调.** `ref_rpo_ess − ref_rpo`:开发 **0/4 格为正**(macro −0.0064)、Ki **0/2**(macro −0.0046)。敏感性扫描把机制钉死——性能对 ESS 单调递增,最优点就在 ESS=1(即完全不加权):

| 设置 | 平均 τ | mbESS 中位数 | macro Δ vs reference |
|---|---:|---:|---:|
| 不加权(`ref_rpo`) | — | 1.000 | **−0.0004** |
| ESS=0.75 | 1.44 | 0.750 | −0.0040 |
| 固定 τ=1 | 1.00 | 0.716 | −0.0039 |
| ESS=0.50(primary) | 1.53 | 0.500 | −0.0062 |
| 固定 τ=5 | 5.00 | 0.265 | −0.0057 |
| ESS=0.25(部分无解) | 38.81 | 0.262 | −0.0258 |
| 固定 τ=10 | 10.00 | 0.176 | −0.0161 |

机制解释:pairwise logistic 的梯度已是 ∂L/∂Δg = −αβ·σ[−β(Δm + αΔg)],**本身就已经聚焦在 reference 排错或接近排错的 pair 上**;再乘一次 σ(−τΔm) 等于重复强调难 pair,同时删掉"不要破坏 reference 已有正确排序"的约束。

> **结论:**Reference-Anchored RPO 是本项目对 RPO 的最后一次方法升级尝试,**闸门四条全不过**。它在六格 macro 上比 reference 低 0.0004、比 `fusion` 高 0.0007——三者在噪声内不可区分。**至此,让原始或改良 RPO 成为最优单一方法的路线全部关闭(第九条独立路线)。**唯一稳定的正向结论是:**OE-Mahalanobis 这个零训练的两样本密度比,是三个 assay endpoint × 两个 backbone 上最好的单一固定检测器**(六格 macro 0.6835,高于 Balanced OOD Head +0.0336、高于原始 RPO +0.0302)。

