# OOD-DPO / RPO 新服务器复现实验手册

本手册按 **2026-09-28 的本机工作树**整理，范围是论文使用的 DrugOOD lbap_general 六格、GOOD HIV/PCBA/ZINC 的 scaffold/size covariate 六格，以及近期 RPO、matched BCE、hinge、OE 对照。Ki、potency、MolRoute 和未进入当前主表的探索性任务不列入必备资源。代码仓库 `https://github.com/langzhouhe/OOD-DPO`，本机工作分支 `revision/oe-acquisition`。仓库根目录 `README.md` 为空，复现时应按本文和冻结协议文件运行。

## 1. 代码、模型、数据与产物的边界

| 内容 | 用途 | 放 GitHub？ |
|---|---|---|
| 根目录 `*.py`、`*.sh`、`data_loader.py`、`utils.py`、`model.py`、冻结协议 Markdown、`environment.yaml`、本手册 | 数据处理、编码、训练、筛选、评估 | 是（文本代码和配置） |
| DrugOOD `drugood_all.zip` 与解包的 `data/raw/lbap_general_*.json` | 六个真实数据集 | 否，新机从官方源下载并校验 |
| GOOD HIV/PCBA/ZINC `data/GOOD*/.../processed/*.pt` | 六个真实数据集 | 否，官方 GOOD 代码下载/处理 |
| MiniMol 1.3.5 自带 `state_dict.pth` | 冻结 512 维分子编码器 | 否，安装包获得 |
| Uni-Mol `weights/mol_pre_no_h_220816.pt` 与 `weights/dict.txt` | 冻结 512 维 3D 编码器 | 否，从官方模型仓库下载 |
| `cache/ood_dpo_cache/`、`repro/` 每 run JSON、`runs/`、`logs/`、`weights/` 下的训练权重 | 特征缓存、结果、检查点 | 否；新机重算 |

只上传代码可以复跑实验，**不能恢复历史 test JSON、训练权重或已编码特征**。本机 `data/` 约 20 GB、`cache/` 约 1.6 GB、`weights/` 约 182 MB、`repro/` 约 483 MB，均不需要放到 GitHub。部分旧 `repro/` 文件已在 Git 跟踪；后续提交要用显式文件列表，不能 `git add -A`。

## 2. 新机目录、Python 与外部 GOOD 代码

本机常用 Python `/root/miniconda3/envs/ood/bin/python`（Python 3.10）。当前实测包版本：`minimol 1.3.5`、`graphium 2.4.7`、`numpy 1.26.4`、`scikit-learn 1.7.2`、`unicore 0.0.1`、`torch 2.5.1+cpu`、`torch-geometric 2.8.0.post1`、`rdkit 2026.3.4`。仓库 `environment.yaml` 锁的是**另一套** CUDA torch 2.5.1/rdkit 2025.03.3 环境；它适合作为依赖清单，不能宣称与当前实装完全相同。MiniMol 编码在 CPU；Uni-Mol 特征预计算脚本可用 GPU。大量 frozen-feature head 实验可以在 CPU 上跑，但 20 worker 的资源开销很大。

新机建议保留 `/root/autodl-tmp/OOD-DPO` 和 `/root/miniconda3/envs/ood` 这两个路径，因为许多 `run_*.sh` 写死它们；纯 Python 入口可直接用所选环境的 `python`。若更换路径，先搜索替换：

```bash
rg -n '/root/autodl-tmp|/root/miniconda3|/home/ubuntu/projects' --glob '*.py' --glob '*.sh' .
```

`data/good_data/good_datasets/good_{hiv,pcba,zinc}.py` 是导入 `GOOD.data.*` 的薄包装。仓库里已有 `GOOD/` 旧代码，但它**没有**当前 loader 要导入的 `GOOD.data` 模块。本机实际由额外克隆的官方 `GOOD_official/`（GOODv1 commit `b53566c9297bc65b90a7f2213fb9ffa930f5b6e5`）提供该模块。新机要单独安装这份公开代码，别把本机 23 MB 的嵌套 Git 克隆误当作已上传内容：

```bash
cd /root/autodl-tmp
git clone -b revision/oe-acquisition https://github.com/langzhouhe/OOD-DPO.git
cd OOD-DPO
git clone --branch GOODv1 https://github.com/divelab/GOOD.git GOOD_official
git -C GOOD_official checkout b53566c9297bc65b90a7f2213fb9ffa930f5b6e5
/root/miniconda3/envs/ood/bin/python -m pip install -e ./GOOD_official
/root/miniconda3/envs/ood/bin/python -m pip install 'minimol==1.3.5' 'graphium==2.4.7' \
  'numpy==1.26.4' 'scikit-learn==1.7.2' 'unicore==0.0.1' gdown
/root/miniconda3/envs/ood/bin/python -c 'from GOOD.data.good_datasets.good_hiv import GOODHIV; from minimol import Minimol; print("imports OK")'
```

安装 PyTorch/PyG/RDKit 时按照新服务器的 CUDA 和 Python 版本选择匹配 wheel；Graphium、MiniMol、Uni-Core/Uni-Mol 的其他依赖可参考 `environment.yaml`。当前仓库也有 `Uni-Core/`、`Uni-Mol/` 代码，但是否需要 `pip install -e` 取决于新机的 `unicore`、`unimol` import；在跑 Uni-Mol 前验证 `from unicore.data import Dictionary; from unimol.models import UniMolModel`。历史流程用了本机 clone 和包的组合，版本改变应记录到实验日志。

## 3. DrugOOD：下载、解包、划分

来源是 [DrugOOD 官方项目](https://drugood.github.io/)公布的 [96 个实现数据集文件夹](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC)。本机 `data/archives/drugood_all.zip` 为作者提供的合集，里面直接有 `drugood_all/lbap_general_{ec50,ic50}_{scaffold,size,assay}.json`。官方文件夹的文件组织可能与本机 zip 不同；新机取得同名官方 JSON 后请按下列目标路径放置，**勿重新按随机规则造原始 DrugOOD 数据**：

```bash
mkdir -p data/raw data/archives
# 若取得 drugood_all.zip：
unzip -j data/archives/drugood_all.zip \
  'drugood_all/lbap_general_ec50_*.json' \
  'drugood_all/lbap_general_ic50_*.json' -d data/raw
ls data/raw/lbap_general_{ec50,ic50}_{scaffold,size,assay}.json
sha256sum data/raw/lbap_general_ec50_assay.json data/raw/lbap_general_ic50_assay.json
```

本机关键哈希（用于判断是否拿到相同原始版本）：

| 文件 | SHA-256 |
|---|---|
| `lbap_general_ec50_assay.json` | `e40d4efef58629aa73720cd824485767821212f59c2f74999c9c51c957e87956` |
| `lbap_general_ic50_assay.json` | `ebd771862ede33f35ee673f5beb629781fdbbd7251cc5bc26d42dcd7a039c315` |
| `lbap_general_ec50_scaffold.json` | `962e94793a0504709108579373fce46ac48df29b2842f43ee0b6df377ded3eda` |
| `lbap_general_ic50_scaffold.json` | `afad93bdacbe3fa0c922579008a8b018c05b42f0b6c195bb8a47a466a184bfca` |

`utils.process_drugood_data()` 从官方 `split.train` 取 ID 训练、`split.iid_val` 取 ID 验证、`split.iid_test` 取 ID 测试、`split.ood_val` 取可用辅助 OOD、`split.ood_test` 取最终 OOD 测试；先校验 SMILES。`data_loader.EnergyDPODataLoader` 再用 `data_seed=42` 和各 split 固定 offset 的 NumPy RNG 无放回抽样，缓存到 `cache/ood_dpo_cache/lbap_general_*_seed42_splits.json`。旧常规表使用 DrugOOD `train_id/train_ood=2000/2000`、验证 `600/600`、测试 `1000/1000`（实际不足时用全部可用数据）。原始 DrugOOD 没有单独的 `val_ood`；loader 会从 `ood_val` 池内切出验证子集。这样按分子切会出现 domain 重叠，**近期修订实验** `revision_run.domain_disjoint_split()` 依据 `domain_id` 再分辅助 OOD 训练/验证，使 domain 集合不交叉。不要把旧版分子随机划分与新 domain-disjoint 结果混为一张可直接比较的表。

## 4. GOOD：自动下载与处理

只用 `GOODHIV`、`GOODPCBA`、`GOODZINC` 的 `scaffold`/`size` + `covariate`，对应六格。官方 GOODv1 的数据类会在首载时从各自 Google Drive 下载并解包，结果应在 `data/GOODHIV/`、`data/GOODPCBA/`、`data/GOODZINC/` 的 `scaffold/processed/` 和 `size/processed/` 下。官方数据类的链接分别在 `GOOD_official/GOOD/data/good_datasets/good_hiv.py`、`good_pcba.py`、`good_zinc.py` 的 `self.url` 中；如果自动下载被 Google Drive 限制，可用 `gdown --fuzzy` 对这些链接手动下载，仍应让官方 loader 处理原始包。

```bash
cd /root/autodl-tmp/OOD-DPO
/root/miniconda3/envs/ood/bin/python - <<'PY'
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_pcba import GOODPCBA
from GOOD.data.good_datasets.good_zinc import GOODZINC
for cls in (GOODHIV, GOODPCBA, GOODZINC):
    for domain in ('scaffold', 'size'):
        for subset in ('train', 'val', 'test', 'id_val', 'id_test'):
            ds = cls(root='./data', domain=domain, shift='covariate', subset=subset)
            print(cls.__name__, domain, subset, len(ds))
PY
```

`utils.process_good_data()` 从官方 train、val OOD、test OOD、id_val、id_test 提取 SMILES；辅助 train-OOD 从 val OOD 分出。旧常规表目标样本量是 GOOD `5000/5000` 训练、`1500/1500` 验证、`2000/2000` 测试。GOOD 的标签提取用于 OE classifier 对照：`good_labels.py` 从已处理 `.pt` 提 SMILES→标签；HIV 用原二值标签，PCBA 使用代码选定的 assay 列，ZINC 以 ID 中位数二值化。这是**本项目对 GOOD 的适配**，不要误称官方 GOOD 的原始 OOD 分类任务。

## 5. 两个预训练编码器与特征缓存

**MiniMol**：官方 [graphcore-research/minimol](https://github.com/graphcore-research/minimol)；`pip install minimol==1.3.5` 安装包内部自带 `ckpts/minimol_v1/state_dict.pth`（本机约 20.8 MB），`model.MinimolEncoder` 调用 `Minimol()` 输出 512 维特征。无需手动下载单独大模型。当前本机 MiniMol 在 CPU 编码，这是正常情况。

**Uni-Mol**：官方 [dptech/Uni-Mol-Models](https://huggingface.co/dptech/Uni-Mol-Models/tree/main) 的 `mol_pre_no_h_220816.pt` 和 `mol.dict.txt`。放到仓库要求的文件名；该 checkpoint 本机哈希 `da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9`，字典本机哈希 `94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127`。

```bash
mkdir -p weights
/root/miniconda3/envs/ood/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
repo = 'dptech/Uni-Mol-Models'
for remote, local in [('mol_pre_no_h_220816.pt','mol_pre_no_h_220816.pt'),
                      ('mol.dict.txt','dict.txt')]:
    src = Path(hf_hub_download(repo, remote))
    Path('weights', local).write_bytes(src.read_bytes())
PY
sha256sum weights/mol_pre_no_h_220816.pt weights/dict.txt
```

`model.py` 在加载字典后**必须补 `[MASK]` token**；否则 `gbf` 的 3D 距离编码权重因 shape 不匹配而保持随机。正确加载应核对 193/193 参数，不能只看到 `strict=False` 无异常就算成功。详见 `RPO_experiment_report.md` E15 和 `ablate_unimol.py`。修改编码器、字典、RDKit 后，旧 pickle 特征缓存必须重建。

先运行旧主流程中**每格一次** `main.py --mode train` 的准备（使用 `--cache_root ./cache --data_seed 42`），使 `cache/ood_dpo_cache/*_seed42_splits.json` 与 MiniMol 特征缓存生成；随后 Uni-Mol 特征用 `precompute_unimol.py` 按格预算。不要直接运行 `run_drugood_backbone.sh` 做第一次缓存生成：它每次并发六格，内存与下载开销大。示例：

```bash
python main.py --mode train --dataset lbap_general_ec50_assay \
  --drugood_subset lbap_general_ec50_assay \
  --data_file ./data/raw/lbap_general_ec50_assay.json \
  --foundation_model minimol --data_seed 42 --cache_root ./cache \
  --debug_dataset_size 100 --epochs 1 --output_dir ./runs/cache_smoke
# 正式缓存请去掉 --debug_dataset_size 100；它会生成不同的 debug 缓存名。
for subset in lbap_general_{ec50,ic50}_{scaffold,size,assay}; do
  python main.py --mode train --dataset "$subset" --drugood_subset "$subset" \
    --data_file "./data/raw/$subset.json" --foundation_model minimol \
    --data_seed 42 --cache_root ./cache --epochs 1 \
    --output_dir "./runs/prepare_$subset"
done
for ds in hiv pcba zinc; do
  for domain in scaffold size; do
    python main.py --mode train --dataset "good_$ds" --good_domain "$domain" \
      --good_shift covariate --data_path ./data --foundation_model minimol \
      --data_seed 42 --cache_root ./cache --epochs 1 \
      --output_dir "./runs/prepare_good_${ds}_${domain}"
  done
done
CUDA_VISIBLE_DEVICES=0 GRAPH_WORKERS=8 UNIMOL_BATCH=128 \
  python precompute_unimol.py
```

正式缓存建议按 `precompute_unimol.py` 的 `CELLS` 列出的 12 格逐一构建，并检查每个 split 的 feature coverage。`revision_run.load()` 在 **train-ID** 上计算每个视图的均值/标准差，构造 MiniMol 或 Uni-Mol 512 维 + Morgan ECFP4 2048 位 + RDKit 10 维描述符；保持 `data_seed=42` 和同样的原始数据才能重现旧 split。缓存可在新机重算，不要上传 pickle。

## 6. 当前论文实验的运行顺序

当前核心公平性对照以 `RPO_OE_TUNING_V2_PROTOCOL.md` 为准：EC50-Assay、IC50-Assay × MiniMol、Uni-Mol 四设定；RPO 与 class-balanced BCE 共用冻结特征、`d→256→128→1` head、同样分子、训练预算和 48 个 recipe。32 个原 Sobol recipe + 16 个预先固定的边界扩展；selection seeds 21–25，final seeds 301–320；每 fit 1,024,000 次 ID+OOD molecule presentations、100 次验证 checkpoint，只用 validation AUROC 选 recipe。`PAIRWISE_HINGE_V2_PROTOCOL.md` 是同设置下的 pairwise hinge 对照。OA baseline `OE_EQUAL48_PROTOCOL.md` 使用各方法 48 点搜索；`OE_EQUAL48_TABLE1_PROTOCOL.md` 把 MSP/ODIN/Energy 三方法扩展到 12 列主表。请先阅读各冻结协议，避免把不同轮次产物拼表。

以一个 assay cell 做阶段化重跑的顺序示例（实际脚本具体参数以 `--help` 为准）：

```bash
# 1. validation-only screen：所有四设定 × selection seeds 21..25
python rpo_opt_v2_screen.py --cell ec50_assay --backbone minimol --seed 21 --device cpu
# 2. 16-point extension 同样在 final 之前完成
python rpo_opt_v2_extension.py --cell ec50_assay --backbone minimol --seed 21 --device cpu
# 3. 对该 cell/backbone 跑齐 selection seeds 21..25 的 screen 和 extension 后：
python rpo_opt_v2_select.py --cell ec50_assay --backbone minimol
#    产生 repro/rpo_opt_v2_selection_*.json，并记录 SHA256
# 4. 检查所有 selection 后，才运行 final seeds 301..320
python rpo_opt_v2_final_one.py --cell ec50_assay --backbone minimol --seed 301 --device cpu
# 5. 汇总及论文表
python rpo_opt_v2_aggregate.py
```

`run_rpo_opt_v2_final.py` 是四设定 × 20 seeds 的 bounded-parallel launcher，默认 20 workers；新机从 `--workers 1` 或 `2` 开始，确认内存占用后再加。对 OE 等额 baseline：

```bash
python run_oe_equal48.py screen --workers 2
python run_oe_equal48.py select --workers 2
# 人工核对、冻结 selection 文件及其 SHA256 后：
python run_oe_equal48.py final --workers 2
python oe_equal48_aggregate.py
# 完整 12-cell Table 1 扩展需加 --scope table1，并依协议限定方法。
```

这些程序写 `repro/*.json` 与 `logs/*.log`。筛选脚本不应读取目标 test；final 在 selection 锁定后才加载 test。以前项目阶段已经看过官方 test 分区，所以近期结果应称 **frozen re-evaluation**，不是全新密封测试。最终核心 endpoint、AUROC/AUPR/常规 FPR95、不同协议结果的来源见 `RPO_experiment_report.md`、`paper/TABLE_DATA_PROVENANCE.md`；不要只依赖旧 `README.md` 或把旧 `main.py` 的 β=0.1/λ=0.01/500 epochs 配方当成新版公平对照。

## 7. 开跑前检查与上传约束

1. `GOOD.data.good_datasets.*`、`minimol`、`unicore`、`unimol` import 成功；DrugOOD 六个 JSON 均存在，至少 assay 两格哈希与本机一致。
2. `weights/dict.txt` 和 Uni-Mol checkpoint 哈希核对；Uni-Mol 193/193 参数加载；同一个 cell 的六份 seed42 split 和两种 512 维特征缓存覆盖齐全。
3. 先跑一个 cell/seed 的 validation screen，再批量跑；冻结 selection JSON 和哈希后才跑 final；聚合时先验证完整 seed 数。额外线程会迅速耗尽内存，`--workers` 依据机器实测调整。
4. GitHub 只提交代码、文本配置、迁移文档；不要使用 `git add -A`，不要提交 `data/`、`cache/`、`weights/`、`GOOD_official/`、结果 JSON、日志、训练权重或论文 PDF。`GOOD_official` 用上文公开 commit 在新机获取即可。
