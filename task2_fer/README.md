# 任务二：FER-2013 面部表情识别（6 类）

## 1. 数据

- 训练 28275 张、测试 7065 张，48×48 灰度（缓存为 `uint8`，训练时双线性放大到 64/96/112）。
- **6 类**（原始 FER-2013 的 Disgust 类样本过少，本赛题数据已去掉）：Angry、Fear、Happy、Neutral、Sad、Surprise。
- 训练集类别计数（按 ImageFolder 排序）：3963 / 4097 / 7192 / 4862 / 3202 / 4959，类别不均衡（Happy 约为 Sad 的 2.2 倍）。
- 指标：准确率；提交 `ID,Emotion`（整数标签）。

## 2. 模型与训练

统一训练脚本 `train_gpu.py`：

| 项目 | 设置 |
|---|---|
| 5 折 | `StratifiedKFold(5, shuffle=True, random_state=42)`，保存每折 OOF logits 与测试 logits |
| 优化器 | AdamW，wd=1e-4，分类头 LR = 主干 ×8，CosineAnnealing 到 1% |
| 损失 | CrossEntropy(label_smoothing=0.1) |
| 混合 | MixUp / CutMix（α=0.3，总概率 0.5，二者各半） |
| 增强 | 随机仿射（±12°、缩放 0.92~1.08、平移 7%、错切 ±8°）、水平翻转、亮度/对比度扰动、高斯模糊、Cutout |
| 推理 | 水平翻转 TTA（logits 平均） |

模型族与 OOF 准确率：

| 族 | 主干 / 尺寸 | OOF |
|---|---|---|
| pseudo | ResNet34@96 + 伪标签再训练 | 0.7083 |
| p2_r34 | 伪标签第二轮 | **0.7192** |
| r34 | ResNet34@96 (CPU 时代) | 0.6990 |
| v2_r34 | R34@112 + EMA + 余弦 50 轮 | 0.7077 |
| r34_96 / r18 / r18_64 | ResNet34@96 / ResNet18@64 / @64 | 0.6958 / 0.6893 / 0.6820 |
| eff_112 | EfficientNet-B0@112 | 0.6868 |
| gnet | GoogLeNet@96 | 0.6410 |

## 3. 集成与近重复覆盖

1. **贪心前向选择（有放回，40 次）**：每次加入使 OOF 准确率提升最大的模型族，取历史最优子集；
2. 最终提交 `submission.csv` 的 OOF 为 **0.7293**，Kaggle 公开榜 **0.736**；
3. **近重复覆盖**：520 张测试图与训练图逐像素完全相同（MD5 与 L2 距离双重验证），另有 5 张距离 < 2.0；留出验证该规则准确率约 **97%**，直接以最近邻训练标签覆盖集成预测（约改动 15 张）。

集成已饱和：等权 0.7187、按族等权 0.7187、贪心 0.7227、去 GoogLeNet 0.7187——继续堆同质模型收益趋零。

## 4. 关键实验结论

- 原环境为 CPU 版 PyTorch（2.14.0+cpu），迁移到 RTX 4070 Laptop 的 CUDA 轮子后训练加速约 30 倍，是所有后续迭代的前提。
- 单折对照：原配方 fold1=0.7022；moderate 增强 + EMA + 60 轮余弦 = 0.7063（EMA 中期可达 0.7089）。
- 人脸对齐（5 点关键点 warp）无收益（OOF 0.6757），因为 48×48 下对齐误差与表情形变同量级。
- 集成收益来源是**架构多样性**而非同族多种子：同族误差相关高，加权几乎不涨。

## 5. 文件

| 文件 | 作用 |
|---|---|
| `train_gpu.py` | 统一 5 折训练（缓存、增强、MixUp/CutMix、OOF） |
| `train_v2.py` | 改进配方：moderate 增强 + EMA + 长轮次 |
| `train_fer.py` | 早期单模型训练 |
| `train_pseudo.py` / `train_pseudo2.py` | 伪标签自训练两轮 |
| `build_submission.py` | 贪心加权 + 近重复覆盖 + 模板对齐 |
| `check_overlap_quality.py` / `find_near_dups.py` | 近重复检测与规则精度验证 |
| `blend_all.py` / `compare_strategies.py` | 等权/按族/贪心/去弱模型对比 |
| `NOTES_PROGRESS.md` | 当时的实验记录与决策依据 |

## 6. 复现

```bash
python train_gpu.py --arch resnet34 --size 96 --epochs 30 --tag r34_96
python train_pseudo.py                          # 用 OOF 生成伪标签再训练
python build_submission.py                      # 贪心集成 + 近重复覆盖
```

数据缓存由 `build_cache.py` 生成（`cache/data.npz`，48×48 uint8）。
