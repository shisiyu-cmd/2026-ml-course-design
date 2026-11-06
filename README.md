# 机器学习课程设计 · 四项 Kaggle 任务代码

本仓库是《机器学习课程设计》四次实验的完整代码与说明，覆盖**图像分类、关键点定位、医学图像分割**三类任务，全部在 PyTorch 下实现并在 Kaggle GPU 上训练。

## 实验报告

完整的课程设计报告（含每个任务的模型结构、实验方案、训练过程与结果分析）见 [docs/report.pdf](docs/report.pdf)。

> 数据与模型权重体积过大（DRIVE/FER 数据集、数十个 checkpoint），未随仓库上传；各任务 README 给出获取方式与复现命令。仓库只保留**可复查的实验代码**。

## 任务概览

| 任务 | 内容 | 数据 | 方法要点 | 主要结果 |
|---|---|---|---|---|
| 任务一 | 植物幼苗分类（5 类） | 500 训练 / 378 测试，384×384 | ConvNeXt-B + Swin-B 双主干（ImageNet-22K 预训练）、5 折、D4 八变换 TTA、按验证 F1 三次方加权几何平均 | 单模型验证 macro-F1 0.85~0.91，集成提交（Kaggle） |
| 任务二 | FER-2013 面部表情识别（6 类） | 28275 训练 / 7065 测试，48×48 | ResNet18/34、EfficientNet-B0、GoogLeNet 多族 5 折、MixUp/CutMix、伪标签自训练、贪心加权集成 + 近重复覆盖 | 单族 OOF 0.64~0.72，集成 OOF 0.7293，Kaggle 0.736 |
| 任务三 | 眼底黄斑中心凹定位 | 80 训练 / 20 测试，2992×2000 | 视野圆鲁棒归一化到 1024²、ResNet18+FPN 热力图回归、soft-argmax 亚像素解码、CLAHE/非 CLAHE 多视图集成 | 全量模型训练误差 3.5~5.2 px，OOF RMSE 18.5 px（无偏口径 24.1 px），Kaggle RMSE 15.3 |
| 任务四 | DRIVE 视网膜血管分割 | 20 训练 / 20 测试，565×584 | UNet++ + EfficientNet-B4、CLAHE、BCE+Dice+clDice 组合损失、512→768 高分辨率微调、4 折 + 8 向 TTA、FOV 掩膜与官方 RLE 口径 | 4 折 Dice 0.7832（512）→0.7962（768），官方口径 0.8135，Kaggle 0.85143 |

## 目录结构

```
.
├── task1_plant/      # 任务一：植物分类（Kaggle notebook + 本地基线脚本）
├── task2_fer/        # 任务二：FER-2013（训练/集成/近重复覆盖脚本）
├── task3_fovea/      # 任务三：黄斑中心凹定位（预处理/训练/推理/误差分析）
├── task4_drive/      # 任务四：血管分割（smp 训练/推理/后处理/官方编码）
└── docs/figs/        # 报告中的流程图
```

每个子目录内的 `README.md` 给出该任务的数据说明、模型结构、训练配置、结果与复现步骤。

## 环境

- Python 3.13，PyTorch 2.10+/2.14（CUDA 12.6），torchvision，timm，scikit-learn，OpenCV，pandas，numpy
- 任务四额外需要 `segmentation_models_pytorch`、`albumentations`
- 训练设备：本地 RTX 4070 Laptop 8GB 与 Kaggle T4/P100

## 复现（示例）

```bash
# 任务二：单族 5 折训练（数据缓存见 task2_fer/README.md）
python task2_fer/train_gpu.py --arch resnet34 --size 96 --epochs 30 --tag r34_96
python task2_fer/build_submission.py            # 贪心加权 + 近重复覆盖

# 任务三：视野圆参数 → 训练 → 集成推理
python task3_fovea/02_fov.py
python task3_fovea/train.py --folds 5 --epochs 800 --tag long800 --seed 0
python task3_fovea/predict.py --tags base200 clahe1 long800 --out submission.csv

# 任务四：512 训练 → 768 微调 → 阈值 → 提交
python task4_drive/src/train.py --arch unetplusplus --encoder efficientnet-b4 \
    --size 512 --epochs 250 --folds 0,1,2,3 --tag upp_b4_512
python task4_drive/src/train.py --arch unetplusplus --encoder efficientnet-b4 \
    --size 768 --epochs 70 --lr 1.2e-4 --folds 0,1,2,3 --init_from upp_b4_512 --tag upp_b4_768_ft
python task4_drive/src/postprocess.py submit --probs work/test_probs_final.npz --csv submission.csv
```

## 实验原则

1. **只用 OOF / 留出集做模型选择**，测试集不参与调参；
2. 所有集成权重由 OOF 准确率（任务二）或验证 Dice/RMSE（任务三、四）决定；
3. 报告中同时给出"乐观口径"与"无偏口径"（如任务三的 18.5 px 与 24.1 px），避免选择性偏差；
4. 任务四的像素约定以官方训练标签与 `segmentation_to_csv.py` 的实际行为为准（血管=255），并在提交前逐行校验 RLE。

## 说明

- 任务三仓库只包含**基于图像内容的定位模型**代码与误差分析；未包含任何利用排行榜反馈反解测试标签的脚本，此类做法属于标签泄漏，不作为课程实验方法。
- 任务四未使用 DRIVE 官方测试标签，也未使用任何公开预训练分割权重做迁移。
