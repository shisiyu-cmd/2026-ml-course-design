# 2D 眼底血管分割 — 解决方案说明

## 1. 任务与数据

- **任务**：2D 眼底图像视网膜血管区域分割。
- **数据**：`train/image` (20 张) + `train/label` (20 张)，`test/image` (20 张)；
  所有图像均为 **565×584 RGB**，与 DRIVE 数据集尺寸一致。
- **提交格式**：游程编码 (RLE)，`Id,Predicted` 两列 CSV。

### ⚠️ 像素值约定（重要）

任务说明中写的是「血管像素 = 0，其他 = 255」，但：

1. 官方提供的 **训练标签实际是 `vessel=255, background=0`**（已用 `np.unique` 核实，仅含 0/255 两个值）；
2. 官方提供的 `segmentation_to_csv.py` 使用 `image[image > 0] = 1` 做二值化，
   即**任何非零像素都算作预测的 mask 像素**。

因此若按说明文字输出 0=血管，则所有背景都会被编码进 RLE，Dice 会趋近于 0。
本项目**遵循训练标签与官方转换脚本的实际约定**：血管 = 255，背景 = 0。
（若评测方确实使用 0=血管，只需在生成预测图后做一次 `255 - img` 反相即可，见第 7 节。）

## 2. 方法概述

| 组件 | 选择 |
| --- | --- |
| 骨干网络 | `UNet++` + `EfficientNet-B4` (ImageNet 预训练)，`segmentation_models_pytorch` |
| 输入尺寸 | 512×512（等比拉伸，标签同步最近邻缩放） |
| 预处理 | LAB 空间 L 通道 CLAHE（clip=2.0, grid=8），保留颜色，增强局部对比 |
| 损失函数 | `0.4·BCE(pos_weight=2) + 0.4·Dice + 0.2·soft-clDice` |
| 优化器 | AdamW，lr=3e-4，wd=1e-4，OneCycleLR，梯度裁剪 5.0 |
| 验证 | 4 折交叉验证（`KFold(shuffle=True, seed=42)`），单张图 Dice |
| 推理 | 4 折模型集成 + 4 向翻转 TTA |
| 后处理 | FOV 掩膜 ∩ 阈值分割（阈值在 OOF 上网格搜索） |

**为什么用 clDice**：血管是细长管状结构，普通 Dice 对细血管（尤其末梢）不敏感。
soft-clDice 直接优化拓扑连通性（中心线召回），能显著改善细小血管的连续性。

## 3. 目录结构

```
segmentation/
├── src/
│   ├── common.py        # 路径、数据集、CLAHE、Dice、RLE 编解码与校验
│   ├── models.py        # 模型工厂 + Dice/BCE/clDice 组合损失
│   ├── train.py         # K 折训练，保存 best 权重与 OOF 概率
│   ├── predict.py       # 多模型/多尺度/TTA 推理
│   ├── postprocess.py   # 阈值网格搜索 + 生成 submission.csv
│   └── bench.py         # 单步速度基准（调试用）
├── work/                # 训练日志、checkpoints、OOF、后处理参数
├── pred_image/          # 预测的二值 PNG（供 segmentation_to_csv.py 使用）
└── submission.csv       # 最终提交文件
```

## 4. 复现步骤

```bash
cd segmentation

# 一键复现（训练 → 微调 → 调阈值 → 推理 → 生成提交）
python src/run_all.py --stage all

# 或者分步执行：
# 1) 训练 4 折（512，约 1.5 小时）
python src/train.py --arch unetplusplus --encoder efficientnet-b4 --size 512 \
    --epochs 250 --min_epochs 120 --patience 45 --batch_size 4 --lr 3e-4 \
    --folds 0,1,2,3 --print_every 10 --tag upp_b4_512

# 2) 768 分辨率微调（从 512 权重初始化，约 1.5 小时）
python src/train.py --arch unetplusplus --encoder efficientnet-b4 --size 768 \
    --epochs 70 --min_epochs 70 --patience 200 --batch_size 2 --lr 1.2e-4 \
    --folds 0,1,2,3 --print_every 5 --tag upp_b4_768_ft --init_from upp_b4_512

# 3) 合并 OOF + 搜索阈值
python src/merge_oof.py --tag upp_b4_768_ft
python src/tune_threshold.py --probs work/oof_upp_b4_768_ft.npz --out work/threshold_768.json

# 4) 测试集推理（4 折集成 + 4 向翻转 TTA，size=768）
python src/predict.py --ckpts work/checkpoints/upp_b4_768_ft_f{0,1,2,3}/best.pth \
    --split test --size 768 --tta i,h,v,hv --out work/test_probs_final.npz

# 5) 生成 submission.csv（内部直接调用官方 turn_to_str 编码）
python src/postprocess.py submit --probs work/test_probs_final.npz \
    --csv submission.csv --image_dir image
```

辅助脚本：

| 脚本 | 作用 |
| --- | --- |
| `src/eval_sweep.py` | 在验证折上扫描推理尺寸 / TTA / 多尺度 |
| `src/tune_ensemble.py` | 多模型加权集成权重网格搜索 |
| `src/val_probs.py` | 为指定推理配置生成验证集概率 |
| `src/visualize.py` | 生成 原图/GT/预测/叠加 对比图 |
| `src/bench.py` | 单步训练速度基准 |

## 5. RLE 格式

`common.rle_encode` 与官方 `segmentation_to_csv.py` **逐位一致**：按行优先展平，
输出 1-based 的 (起始位置, 长度) 对。`common.check_rle` 校验：

- 像素对数量为偶数；
- 起始位置与长度均为正数；
- 起始位置严格递增且不与前一段重叠（不重复编码同一像素）；
- 不越界。

提交前对每一张图都执行了该校验。

## 6. 结果

### 4 折交叉验证（UNet++ / EfficientNet-B4，512 训练，Dice @ 0.5）

| Fold | 验证图像 | 512 训练 | **768 微调（最终）** |
| --- | --- | --- | --- |
| 0 | 1, 2, 9, 16, 18 | 0.7987 | **0.8100** |
| 1 | 4, 6, 12, 17, 19 | 0.7565 | **0.7661** |
| 2 | 3, 5, 10, 14, 20 | 0.7945 | **0.8071** |
| 3 | 7, 8, 11, 13, 15 | 0.7832 | **0.8017** |
| **均值** | | **0.7832** | **0.7962** |

Fold 1 偏低是该折验证图像本身更难（训练曲线稳定在 ~0.76，非欠拟合）。

**分辨率是关键**：512 训练时标签被下采样，细血管丢失；用 768 微调（从 512 权重初始化，
低学习率 1.2e-4，70 epoch）在**每一折都提升 +0.007 ~ +0.018**，整体 **+1.3 Dice**。

### 推理配置搜索（512 权重，仅改推理设置）

| 推理设置 | Val Dice |
| --- | --- |
| 512 + TTA(i,h,v,hv) | 0.7957 |
| **640 + TTA(i,h,v,hv)** | **0.8025** |
| 640 + TTA + 多尺度(0.85,1.15) | 0.8017 |

推理放大到 640 带来约 **+0.7 Dice**；多尺度无额外收益。

### 集成

| 方案 | Val Dice |
| --- | --- |
| UNet++/EffB4 单模型 (640) | 0.8014 |
| + UNet/ResNet34 (权重 0.78/0.22) | 0.8020 |
| UNet/ResNet34 单模型 (640) | 0.7939 |

ResNet34 模型偏弱，加权集成仅 +0.0006（在噪声范围内），因此**最终提交以 768 微调的
UNet++ 4 折集成为主**，不引入拖后腿的模型。


## 7. 提交文件

- `submission.csv`：20 行，`Id,Predicted`，Id 按 1~20 自然排序。
- 已用官方 `segmentation_to_csv.py` 重新生成并逐行比对，**完全一致**。
- 每行 RLE 均通过校验：像素对数量为偶数、起始位置与长度为正、严格递增无重叠、不越界。

## 8. 关于像素约定的备注

若评测方严格按照任务说明「血管=0」，只需在 `postprocess.py` 写出 PNG 时反相：

```python
img = ((1 - m) * 255).astype(np.uint8)   # 代替 (m * 255)
```

由于 `segmentation_to_csv.py` 会把所有非零像素当作 mask，此时 RLE 编码的是背景区域，
因此该反相必须与评测方的解码方式配套。本项目默认采用与训练标签一致的 255=血管。

## 9. 提交格式的关键细节（踩坑记录）

`segmentation_to_csv.py` 会**先把预测图 resize 到 512×512（PIL BILINEAR）再二值化编码**。
因此 RLE 的下标是基于 **512×512 网格** 的，而不是原图 565×584。

本项目的 `postprocess.py submit` 直接调用官方 `segmentation_to_csv.turn_to_str` 生成 RLE，
已与「先跑官方脚本」的输出逐字节比对一致（20/20 行完全相同），避免了下标错位。

其他工程细节（Windows 环境）：

- `cv2.imread/imwrite` 在含中文的路径下会**静默失败**，已改用 `np.fromfile + cv2.imdecode`
  与 `cv2.imencode + tofile` 封装（`common.imread_unicode / imwrite_unicode`）。
- `channels_last` 在该 GPU 上让训练慢 100 倍，已默认关闭。
- DataLoader 使用 `num_workers=0`（本机 pagefile 仅 2 GB，spawn 子进程加载 torch DLL 会失败）。

## 10. 结果小结

| 项目 | Dice |
| --- | --- |
| 512 训练 4 折均值（原图分辨率） | 0.7832 |
| **768 微调 4 折均值（原图分辨率）** | **0.7962** |
| 单折最佳（fold 0） | 0.8100 |
| **按官方编码口径评测（512 BILINEAR + >0）** | **0.8135** |

> ⚠️ 注意两个口径的差异：`segmentation_to_csv.py` 会先把 mask 用 PIL BILINEAR 缩放到
> 512×512 再取 `>0`，这个「双线性 + 非零即前景」的过程对预测和 GT 同时做了轻微膨胀，
> 使同一批预测的 Dice 从原图的 0.796 变为 **0.814**。因此我把阈值直接放在
> **官方口径**下搜索（`src/tune_threshold_official.py`），thr=0.48~0.50 为平台最优，
> 取 0.50 更稳健。排行榜上的分数预期落在 **0.81 左右**。

对比 DRIVE 公开榜单，UNet++/EfficientNet-B4 + 高分辨率微调处于有竞争力的水平。
进一步可尝试：更大的推理尺寸（896/1024）、SegFormer-B3 等 Transformer 骨干、
以及伪标签半监督（用测试集 20 张图做自训练）。



