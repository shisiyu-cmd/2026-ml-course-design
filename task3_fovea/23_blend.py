# -*- coding: utf-8 -*-
"""按权重融合若干份已预测好的结果，生成提交文件。

好处: 不需要重新跑模型。predict.py 只支持等权，而 allF(训练误差1.15)
明显强于 allA-D(3.5-5.2)，需要能调权重。

用法:
  python 23_blend.py --in runs/test_pred_allF.csv:2 runs/test_pred_all4.csv:1 --out submission.csv
"""
import os, argparse
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inputs", nargs="+", required=True,
               help="路径:权重，例如 runs/a.csv:2")
p.add_argument("--out", default="submission.csv")
p.add_argument("--dump", default=None, help="逐图预测输出路径")
a = p.parse_args()

specs = []
for s in a.inputs:
    path, _, w = s.rpartition(":")
    if not path:                 # 没有冒号 -> 默认权重 1
        path, w = s, "1"
    specs.append((path, float(w)))

meta = C.load_meta()
ref = None
acc = None
wsum = 0.0
rows_meta = None
for path, w in specs:
    d = pd.read_csv(path).set_index("key")
    if ref is None:
        ref = list(d.index)
        acc = np.zeros((len(d), 2))
        rows_meta = d
    assert list(d.index) == ref, "图片顺序不一致: %s" % path
    # 必须在**原图坐标**上融合: canvas_x/canvas_y 依赖各自的画布边长，
    # 混用 768 与 1024 画布的模型时直接平均 canvas 坐标会算错 1.333 倍。
    acc += w * np.stack([d.x.values, d.y.values], 1)
    wsum += w
    print("  + %-38s 权重 %.2f  (峰值 %.3f)" % (path, w, d.peak.mean()))
acc /= wsum
print()

# 已直接得到原图坐标，仅反算画布坐标用于展示
rows = []
for i, k in enumerate(ref):
    x, y = float(acc[i, 0]), float(acc[i, 1])
    md = meta[k]
    cx, cy = C.to_canvas(md, x, y)
    rows.append(dict(key=k, keynum=int(k.split("/")[1]), x=x, y=y,
                     canvas_x=cx, canvas_y=cy,
                     peak=float(rows_meta.peak.values[i]),
                     dist_r=float(np.hypot(x - md["cx"], y - md["cy"]) / md["r"]),
                     W=md["W"]))
df = pd.DataFrame(rows).sort_values("keynum")

# 与参考（第一份输入）比较
d0 = pd.read_csv(specs[0][0]).set_index("key")
diff = np.hypot(df.set_index("key").x - d0.x, df.set_index("key").y - d0.y)
print("融合结果 vs 第一份输入: 最大差 %.1f px, 中位差 %.1f px" % (diff.max(), diff.median()))
if diff.max() > 2:
    i = int(np.argmax(diff.values))
    print("  差异最大: %s (%.1f px)" % (diff.index[i], diff.values[i]))
print()

if a.dump:
    os.makedirs(os.path.dirname(a.dump) or ".", exist_ok=True)
    df.to_csv(a.dump, index=False)
    print("已保存逐图预测:", a.dump)

sub = pd.read_csv(os.path.join(C.ROOT, "sample_submission.csv"))
pred = {}
for r in df.itertuples():
    pred["%d_Fovea_X" % r.keynum] = r.x
    pred["%d_Fovea_Y" % r.keynum] = r.y
missing = [i for i in sub.ImageID if i not in pred]
if missing:
    raise SystemExit("缺少条目: %s" % missing[:5])
sub["value"] = sub.ImageID.map(pred)
assert sub.value.notna().all()
assert sub.ImageID.tolist() == pd.read_csv(
    os.path.join(C.ROOT, "sample_submission.csv")).ImageID.tolist()
sub.to_csv(a.out, index=False)
print("已写出提交: %s (%d 行, 格式与 sample_submission.csv 完全一致)" % (a.out, len(sub)))