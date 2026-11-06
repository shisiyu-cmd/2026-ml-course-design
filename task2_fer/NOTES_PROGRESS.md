# FER-2013 任务进展记录（2026-10-17）

## 当前提交状态

| 文件 | OOF 准确率 | 说明 |
|---|---|---|
| `submission_prior_736_backup.csv` | ~0.722 | 你原来的 22 模型等权 + 近重复覆盖，Kaggle **73.6%** |
| `submission.csv`（当前） | **0.7242** | 我的贪心加权混合 + 近重复覆盖，**未提交验证** |

⚠️ 当前 `submission.csv` 的 OOF 只比原来高 0.15%，**很可能在 Kaggle 上是噪声级别的差异**。
如果你想稳妥，随时可回滚：`Copy-Item submission_prior_736_backup.csv submission.csv -Force`

## 今天解决的核心问题

**你之前的模型全是在 CPU 上训练的**（torch 是 `2.14.0+cpu`）。机器其实有 **RTX 4070 Laptop (8GB)**。
现已装好 CUDA 版 PyTorch（`2.14.0+cu126`），训练速度提升约 30 倍，这是后续所有改进的前提。

安装方式（网络限速，必须用多线程分段下载 + 本地 wheel 安装）：
- `pdl.py` — 16 线程分段下载器（单连接仅 0.1 MB/s，聚合可达 2.6 MB/s）
- wheel 源：`https://mirrors.aliyun.com/pytorch-wheels/cu126/`
- 已下载的 wheel 在 `%TEMP%\whl\`（若已清理需重新下载，约 18 分钟）

## 关键结论（用 OOF 诚实验证，不再靠猜）

我为你**全部 22 个旧 checkpoint 生成了 OOF 预测**（`prior_oof.py` / `prior_oof_all.py`）。
已验证：你旧脚本的 `StratifiedKFold(5, shuffle=True, random_state=42)` 与我的折划分**完全一致**
（r34 fold1 在我 fold1 验证集上 = 0.7022，与你记录的 0.6937 差异来自 TTA）。

各模型族 OOF 准确率：
```
pseudo (伪标签 R34)  0.7083   ← 最强，印证你的判断
r34                  0.6990
r34_96 (新,GPU)      0.6958
eff                  0.6912   (仅 2/5 折)
r18                  0.6893
r18_64 (新,GPU)      0.6820
align                0.6757   (仅 4/5 折，确实没用)
gnet (GoogLeNet)     0.6410   ← 明显最弱，但贪心仍选了它
```

**集成已饱和**：等权 0.7187 / 按族等权 0.7187 / 贪心 0.7227 / 去掉 gnet 0.7187。
用一半数据选权重、另一半验证 = 0.7224，说明**没有过拟合，就是到顶了**。
→ 这印证了你的经验：**加模型、调权重、多 TTA、对齐都无效**。根因是模型间相关性太高。

## 唯一还有空间的方向：单模型能力

单折对照实验（`exp_fold.py`，fold1，与 r34 基线 0.7022 对比）：

| 配方 | fold1 准确率 |
|---|---|
| 你原来（30ep, RandAugment mag9） | 0.7022 |
| **moderate 增强 + EMA + 60ep 余弦** | **0.7063** ✅ |

EMA（指数移动平均）是关键，中期能到 0.7089。提升不大（+0.4%），但是**真实且可叠加**的。

## 明天从这里继续

v2 训练在 fold1 ep30 时被中断（当时 EMA 0.7089），**没有保存任何产物**。重跑：

```powershell
cd C:\Users\古恩宇\Downloads\2026task2
python train_v2.py --arch r34 --size 112 --epochs 50 --bs 96 --lr 2e-4 --tag v2_r34
```
每折约 30 分钟，5 折约 2.5 小时。

跑完后生成提交（会自动把 v2 纳入贪心选择）：
```powershell
python build_submission.py v2_r34
```

若还想再提升，建议按**架构多样性**而非同族堆叠，优先级：
1. `--arch dense`（DenseNet121，结构差异最大）
2. `--arch mbv3`（MobileNetV3-Large）
3. 伪标签第二轮（用当前集成结果当伪标签，你最有效的手段）

## 重要文件说明

| 文件 | 作用 |
|---|---|
| `train_v2.py` | 改进配方 5 折训练（EMA+moderate aug）|
| `exp_fold.py` | 单折快速配方实验，用来试新想法 |
| `build_submission.py` | 贪心选权 + 近重复覆盖 + 按模板输出 |
| `prior_oof_all.py` | 为旧 checkpoint 生成 OOF（已跑完，结果在 `prior_preds/`）|
| `blend_all.py` / `compare_strategies.py` | 集成策略对比分析 |
| `cache/data.npz` | 48×48 图像缓存（28k 训练 + 7065 测试）|

## 近重复覆盖（已验证可靠，建议保留）

- 520 张测试图与训练集**逐像素完全相同**（我用距离验证，与 MD5 结果 100% 吻合）
- 另有 5 张 dist<2.0 的近似重复
- 留出集验证：该规则准确率约 **97%**（`check_overlap_quality.py`）
- 覆盖会改动约 15 张预测，是稳定的小幅收益
