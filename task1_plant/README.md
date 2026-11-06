# 任务一：植物幼苗图像分类

## 1. 任务与数据

Kaggle 竞赛 `2026task1`（目录名 `dataset-for-task2`），按 `ID,Category` 提交。

| 项目 | 说明 |
|---|---|
| 类别 | 5 类：Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet |
| 训练集 | 500 张（每类 100 张，图像为田间幼苗，背景含土壤、砾石、标尺与二维码） |
| 测试集 | 378 张 |
| 指标 | macro-F1 |
| 输入 | 224（基线）/ 320（v5）/ 384（v6、v7） |

## 2. 方案迭代

| 版本 | 主干 | 关键改动 | 结果 |
|---|---|---|---|
| v1 基线 | ResNet50 ×3 seed | 224，头部 3 轮 + 全网 15 轮，3 尺寸 × 翻转 TTA | 单 seed 验证 macro-F1 0.8518 / 0.9121（15% 留出集，75 张，方差大）；原方案记录 0.9021 |
| v3 集成 | ResNet50 + EfficientNetV2-S + DenseNet121 | 320，两阶段差异化 LR，Stratified 85/15，MixUp + 类别平衡采样，6 形态 TTA | 验证集三模型集成 F1 |
| v5 | EfficientNetV2-M + ConvNeXt-Base | ExG 超绿指数 + Otsu 分割植株前景，384，5 折 × 2 主干（10 模型），4 形态 TTA | OOF 选权重 |
| v6 | ConvNeXt-B + EfficientNetV2-M | 全量 500 张训练（不设验证折）、每架构 2 seed、头部 3 + 全网 18 轮 | 应对 5 折下每折仅 400 张导致的欠拟合 |
| **v7 最终** | **ConvNeXt-B + Swin-B（ImageNet-22K→1K，384）** | 5 折 × 2 主干，12 轮（1 头部 + 11 全网），D4 八变换 TTA，权重 =（Val F1）³ × 架构系数，对数概率几何平均 | 见 `plant_classification_v7.py` |

## 3. 最终配方（v7）

- **数据增强**：RandomResizedCrop(384, scale=0.7~1.0, ratio=0.85~1.15)、水平/垂直翻转、旋转 ±180°、ColorJitter(0.1)。
- **训练**：AMP；batch=8 + 梯度累积 2（等效 16）；头部 1 轮 AdamW lr=1e-3；全网 11 轮 AdamW，ConvNeXt lr=1e-4、Swin lr=5e-5，wd=1e-2，线性 warmup 1 轮后余弦退火到 1%，label smoothing=0.03。
- **推理**：D4 对称群 8 种变换（0/90/180/270° 旋转 × 翻转）分别前向，取 log-softmax 平均；模型权重 `(best_val_f1)^3 × mult`（ConvNeXt 1.0、Swin 1.5），归一化后加权求和，取 argmax。
- **提交校验**：ID 全覆盖、类别合法、行数与模板一致三重断言。

## 4. 文件

| 文件 | 作用 |
|---|---|
| `plant_classification_v7.py` | 最终方案（Kaggle 脚本，可直接粘贴为 notebook） |
| `plant_classification_v6.ipynb` | 全量数据 + 多 seed 训练 |
| `plant_classification_v5.ipynb` | ExG 前景分割 + EffV2-M/ConvNeXt-B |
| `plant_classification_cv5.ipynb` | DenseNet121/EffV2-S/ConvNeXt-T 三架构 5 折 |
| `plant_classification_ensemble.ipynb` | 三架构集成（320） |
| `train.py` / `predict.py` | 本地 ResNet50 / EfficientNet-B0 基线（含两阶段训练与 TTA） |

## 5. 复现

`v7` 依赖 Kaggle 数据集挂载（`/kaggle/input/**/train`），在 Kaggle Notebook 中设置 GPU + Internet 后直接运行；本地运行需修改 `DATA_ROOT` 并将 `torch`、`timm` 与预训练权重准备好。
