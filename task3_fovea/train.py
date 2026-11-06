# -*- coding: utf-8 -*-
"""中心凹热力图回归 —— K 折交叉验证训练

loss = MSE(sigmoid(heatmap), 高斯目标) + w_xy * L1(soft-argmax 坐标, 真值坐标)

用法:
  python train.py --folds 5 --epochs 150 --backbone resnet18
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--w_xy", type=float, default=5.0)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max_rot", type=float, default=20.0)
    p.add_argument("--scale_lo", type=float, default=0.90)
    p.add_argument("--scale_hi", type=float, default=1.10)
    p.add_argument("--trans", type=float, default=0.04)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--mid", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="exp")
    p.add_argument("--only_fold", type=int, default=-1)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--clahe", action="store_true", help="预处理加 CLAHE 局部对比度增强")
    p.add_argument("--clahe_clip", type=float, default=3.0)
    p.add_argument("--subset", type=int, default=0,
                   help=">0 时只用前 N 张训练（小样本过拟合测试用）")
    p.add_argument("--no_aug", action="store_true", help="关闭数据增强（小样本过拟合测试用）")
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--eval_train", action="store_true", help="同时报告训练集 RMSE")
    p.add_argument("--select", default="val", choices=["val", "train"],
                   help="按哪个指标挑选最佳权重")
    p.add_argument("--canvas", type=int, default=1024, help="画布边长（影响精度上限）")
    p.add_argument("--workers", type=int, default=3,
                   help="DataLoader 进程数（把增强从 GPU 关键路径移开）")
    p.add_argument("--loss_xy", default="l1", choices=["l1", "mse"],
                   help="坐标损失。指标是 RMSE(平方损失)，L1 会忽略大误差 -> 用 mse 更匹配")
    p.add_argument("--all_data", action="store_true",
                   help="用全部 80 张训练（不做折划分）。无 OOF，仅用于最终提交的模型")
    p.add_argument("--snap_every", type=int, default=0,
                   help=">0 时每 N 轮存一次快照（全量模型无验证集，可用快照集成代替选轮次）")
    p.add_argument("--ecc_w", type=float, default=0.0,
                   help="按中心凹离心度过采样的强度：权重 = 1 + ecc_w * (ecc > thr)")
    p.add_argument("--ecc_thr", type=float, default=0.20, help="过采样的离心度阈值")
    p.add_argument("--box", action="store_true",
                   help="加黄斑框回归头（赛题提示的两阶段路线）。框中心与坐标真值差 2.4px")
    p.add_argument("--w_box", type=float, default=3.0, help="框损失的权重")
    p.add_argument("--ext", action="store_true",
                   help="把 ext/ 下的外部数据集并入**训练**（不并入验证，保证 val RMSE 可比）")
    p.add_argument("--outdir", default="runs")
    return p.parse_args()


def build_model(args):
    m = C.FoveaNet(args.backbone, pretrained=True, drop=args.drop, mid=args.mid,
                   box_head=getattr(args, "box", False))
    return m


def make_optimizer(model, args):
    head, back = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (back if n.startswith("enc.") else head).append(p)
    return torch.optim.AdamW([
        {"params": back, "lr": args.lr * args.backbone_lr_mult},
        {"params": head, "lr": args.lr},
    ], weight_decay=args.wd)


def lr_at(epoch, args):
    if epoch < args.warmup:
        return (epoch + 1) / max(1, args.warmup)
    t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, loader, device, beta, use_tta=True):
    """返回 (soft预测, hard预测, 框预测或None) —— 均为原图像素坐标。"""
    model.eval()
    has_box = getattr(model, "box_head", None) is not None
    sp, hp, ks, orig, bp = [], [], [], [], []
    for b in loader:
        img = C.normalize_batch(b["img"].to(device, non_blocking=True))
        if has_box:
            out, bx = model(img, return_box=True)
        else:
            out = model(img)
        prob = torch.sigmoid(out)
        s = C.soft_argmax(prob, beta)
        h = C.hard_argmax_refine(prob)
        bxy = C.box_decode(bx.float(), C.CANVAS).cpu().numpy() if has_box else None
        if use_tta:
            if has_box:
                out_f, bxf = model(torch.flip(img, dims=[3]), return_box=True)
            else:
                out_f = model(torch.flip(img, dims=[3]))
            prob_f = torch.sigmoid(out_f)
            # 水平翻转下, 画布坐标 x -> S-1-x。换算到热力图格子坐标(格子 j 对应画布 4j,
            # 而画布 S-1 对应格子 (S-1)/4 = h-0.25), 故映射为 (h-0.25) - x。
            off = prob.shape[3] - 0.25
            sf = C.soft_argmax(prob_f, beta)
            sf[:, 0] = off - sf[:, 0]
            hf = C.hard_argmax_refine(prob_f)
            hf[:, 0] = off - hf[:, 0]
            s = (s + sf) / 2
            h = (h + hf) / 2
            if has_box:
                bxyf = C.box_decode(bxf.float(), C.CANVAS).cpu().numpy()
                bxyf[:, 0] = (C.CANVAS - 1) - bxyf[:, 0]
                bxy = (bxy + bxyf) / 2
        sp.append(s.cpu().numpy()); hp.append(h.cpu().numpy())
        if has_box:
            bp.append(bxy)
        ks += b["key"]; orig.append(b["orig"])
    sp = np.concatenate(sp); hp = np.concatenate(hp); orig = np.concatenate(orig)
    bp = np.concatenate(bp) if has_box else None

    def back(a, is_cell=True):
        """热力图格子坐标(is_cell=True) 或画布坐标 -> 原图像素坐标。"""
        canvas = a * C.STRIDE if is_cell else a
        out = np.empty_like(canvas)
        for i, k in enumerate(ks):
            out[i, 0], out[i, 1] = C.to_orig(META[k], canvas[i, 0], canvas[i, 1])
        return out
    return back(sp), back(hp), (back(bp, False) if bp is not None else None), ks, orig


def run_fold(args, fold, tr_keys, va_keys, cache, gt_map, device, log,
             force_final=False):
    C.set_seed(args.seed * 1000 + fold)
    model = build_model(args).to(device)
    opt = make_optimizer(model, args)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_at(e, args))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    w = args.workers
    # workers>0 时缓存不随 Dataset pickle，由各 worker 自行惰性构建
    ck = cache if w == 0 else None
    box_map = C.load_box_map(META) if getattr(args, "box", False) else None
    ds_tr = C.FoveaDataset(tr_keys, ck, META, gt_map, True, args.sigma,
                           seed=args.seed * 100 + fold, max_rot=args.max_rot,
                           scale=(args.scale_lo, args.scale_hi), trans=args.trans,
                           no_aug=args.no_aug, clahe=args.clahe,
                           clahe_clip=args.clahe_clip, box=box_map)
    ds_va = C.FoveaDataset(va_keys, ck, META, gt_map, False, args.sigma,
                           clahe=args.clahe, clahe_clip=args.clahe_clip, box=box_map)
    ds_te = C.FoveaDataset(tr_keys, ck, META, gt_map, False, args.sigma,
                           clahe=args.clahe, clahe_clip=args.clahe_clip, box=box_map) \
        if args.eval_train else None
    coll = C.HeatmapCollate(args.sigma, "cpu")
    # 按中心凹离心度过采样：测试集偏向周边(ecc 中位 0.22 vs 训练 0.12)，
    # 而模型在周边组的 p90 误差是近中心组的 1.4 倍。过采样保持位置先验不变，
    # 只重新平衡样本分布（对比之下，加大平移增强会破坏位置先验，已验证更差）。
    if args.ecc_w > 0:
        eccs = np.array([np.hypot((gt_map[k][0] - META[k]["cx"]) / META[k]["r"],
                                  (gt_map[k][1] - META[k]["cy"]) / META[k]["r"])
                         for k in tr_keys])
        wts = 1.0 + args.ecc_w * (eccs > args.ecc_thr)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(wts, dtype=torch.double), num_samples=len(tr_keys), replacement=True)
        log("  离心度过采样: %d/%d 张高离心图被加权 %.1fx (thr=%.2f)"
            % (int((eccs > args.ecc_thr).sum()), len(eccs), 1 + args.ecc_w, args.ecc_thr))
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                                            num_workers=w, collate_fn=coll, drop_last=False,
                                            persistent_workers=w > 0,
                                            prefetch_factor=4 if w > 0 else None)
    else:
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                                            num_workers=w, collate_fn=coll, drop_last=False,
                                            persistent_workers=w > 0,
                                            prefetch_factor=4 if w > 0 else None)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                                        num_workers=w, collate_fn=coll,
                                        persistent_workers=w > 0,
                                        prefetch_factor=4 if w > 0 else None)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.bs, shuffle=False,
                                        num_workers=w, collate_fn=coll,
                                        persistent_workers=w > 0,
                                        prefetch_factor=4 if w > 0 else None) if ds_te else None
    hs = C.CANVAS // C.STRIDE
    best = 1e9
    best_state = None
    for ep in range(args.epochs):
        model.train()
        tot, nb = None, 0
        t_data = t_step = 0.0
        t0 = time.time()
        for b in dl_tr:
            img = C.normalize_batch(b["img"].to(device, non_blocking=True))
            tgt = b["hm"].to(device, non_blocking=True)
            gt_canvas = b["canvas"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time(); t_data += t1 - t0
            with torch.amp.autocast("cuda", enabled=True):
                if box_map is not None:
                    out, bx = model(img, return_box=True)
                else:
                    out = model(img)
                prob = torch.sigmoid(out)
                loss_hm = F.mse_loss(prob.float(), tgt.float())
                coord = C.soft_argmax(prob.float(), args.beta) * C.STRIDE
                if args.loss_xy == "mse":
                    # 指标是 RMSE(平方损失)，L1 会忽略大误差；用 MSE 才对尾部负责
                    loss_xy = F.mse_loss(coord / C.CANVAS, gt_canvas / C.CANVAS)
                else:
                    loss_xy = F.l1_loss(coord / C.CANVAS, gt_canvas / C.CANVAS)
                loss = loss_hm + args.w_xy * loss_xy
                if box_map is not None:
                    loss_box = C.box_loss(bx.float(), b["box"].to(device))
                    loss = loss + args.w_box * loss_box
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            # 累加在 GPU 上，避免每次迭代都做 device 同步（float(loss) 会打断流水线）
            tot = loss.detach() if tot is None else tot + loss.detach()
            nb += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time(); t_step += t0 - t1
        sched.step()
        if args.snap_every > 0 and (ep + 1) % args.snap_every == 0:
            torch.save({"state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()},
                        "args": vars(args), "fold": fold, "epoch": ep + 1},
                       os.path.join(args.outdir, "%s_snap%04d.pt" % (args.tag, ep + 1)))
        if args.profile:
            log("      [剖析] 取数 %.2fs  计算 %.2fs  合计 %.2fs/epoch"
                % (t_data, t_step, t_data + t_step))
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            sp, hp, bpred, ks, orig = evaluate(model, dl_va, device, args.beta)
            r_soft = C.report_rmse(sp, orig, ks, verbose=False)
            r_hard = C.report_rmse(hp, orig, ks, verbose=False)
            msg = ("  [%s f%d] ep%4d loss=%.4f  val RMSE soft=%.2f hard=%.2f"
                   % (args.tag, fold, ep + 1, float(tot) / max(nb, 1), r_soft, r_hard))
            sel = r_soft
            if bpred is not None:
                # 框中心预测误差（真值是点坐标而非框中心，故有约 2.4px 的固有下限）
                msg += "  box=%.2f" % C.report_rmse(bpred, orig, ks, verbose=False)
            if dl_te is not None:
                tp, _, _, _, torig = evaluate(model, dl_te, device, args.beta, use_tta=False)
                r_tr = C.report_rmse(tp, torig, None, verbose=False)
                msg += "  | 训练 RMSE=%.2f" % r_tr
                if args.select == "train":
                    sel = r_tr
            log(msg)
            if sel < best:
                best = sel
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not force_final and best_state is not None:
        model.load_state_dict(best_state)
    sp, hp, bpred, ks, orig = evaluate(model, dl_va, device, args.beta)
    return model, sp, hp, bpred, ks, orig, best


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.outdir, "%s.log" % args.tag), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    log("=" * 78)
    log("实验 %s | %s" % (args.tag, json.dumps(vars(args), ensure_ascii=False)))

    C.set_canvas(args.canvas)          # 必须在建缓存之前
    cache = C.build_cache(META, clahe=args.clahe, clahe_clip=args.clahe_clip)
    # CLAHE 会改变画布分布，统计量必须按当前缓存重算并存入权重文件。
    # 关键: 有外部数据时**只用本赛题数据**算统计量。否则 362 张 IDRiD(亮度+25%、
    # 对比度-34%) 会把归一化常数拉偏，使测试图的预处理与本赛题历史模型不一致。
    stat_src = cache
    if args.ext:
        stat_src = {k: v for k, v in cache.items() if not k.startswith("ext/")}
        log("归一化统计量只在本赛题 %d 张上计算（外部数据 %d 张不参与）"
            % (len(stat_src), len(cache) - len(stat_src)))
    mean, std = C.cache_stats(stat_src)
    C.set_norm(mean, std)
    args.norm_mean, args.norm_std = list(mean), list(std)
    log("归一化统计量: mean=%s std=%s" % (np.round(mean, 4).tolist(),
                                         np.round(std, 4).tolist()))
    gt = C.load_gt()
    gt_map = C.load_gt_map(gt)
    keys = sorted(gt_map.keys())
    strat = np.array([0 if META[k]["W"] < 2500 else 1 for k in keys])
    # 外部数据集：只并入训练，不并入验证 —— 这样 val / 训练 RMSE 与历史实验可比
    ext_keys = []
    if args.ext:
        _, ext_gt = C.load_ext()
        ext_keys = sorted(ext_gt.keys())
        if not ext_keys:
            raise SystemExit("--ext 已开启但 ext/ 为空，请先运行 37_build_ext.py")
        gt_map.update(ext_gt)
        log("外部数据: %d 张并入训练（训练集从 %d 增至 %d）"
            % (len(ext_keys), len(keys), len(keys) + len(ext_keys)))

    all_sp, all_hp, all_orig, all_keys = [], [], [], []
    t0 = time.time()
    if args.all_data:
        # 用全部 80 张(+外部)训练。没有留出集，故不选权重，直接取余弦退火终点模型。
        # 验证仍只用赛题数据，以便与历史实验的样本内误差可比。
        model, sp, hp, bpred, ks, orig, _ = run_fold(
            args, 0, keys + ext_keys, keys, cache, gt_map, device, log, force_final=True)
        torch.save({"state": model.state_dict(), "args": vars(args), "fold": 0,
                    "all_data": True},
                   os.path.join(args.outdir, "%s_all.pt" % args.tag))
        log("已保存全量模型: %s_all.pt" % args.tag)
        r = C.report_rmse(sp, orig, ks, verbose=False)
        eu = np.sqrt(((sp - orig) ** 2).sum(1))
        log("  热力图 soft: RMSE=%.2f | 均值=%.2f 中位=%.2f p90=%.2f 最大=%.2f"
            % (r, eu.mean(), np.median(eu), np.percentile(eu, 90), eu.max()))
        if bpred is not None:
            rb = C.report_rmse(bpred, orig, ks, verbose=False)
            eub = np.sqrt(((bpred - orig) ** 2).sum(1))
            log("  框中心    : RMSE=%.2f | 均值=%.2f 中位=%.2f p90=%.2f 最大=%.2f"
                % (rb, eub.mean(), np.median(eub), np.percentile(eub, 90), eub.max()))
        logf.close()
        return
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=100 + rep)
        for fold, (ti, vi) in enumerate(skf.split(keys, strat)):
            if args.only_fold >= 0 and fold != args.only_fold:
                continue
            tr = [keys[i] for i in ti]; va = [keys[i] for i in vi]
            if args.subset > 0:
                tr = tr[:args.subset]
            tr = tr + ext_keys          # 外部数据每折都全量并入训练
            model, sp, hp, bpred, ks, orig, best = run_fold(args, fold, tr, va, cache, gt_map, device, log)
            all_sp.append(sp); all_hp.append(hp); all_orig.append(orig); all_keys += ks
            torch.save({"state": model.state_dict(), "args": vars(args), "fold": fold},
                       os.path.join(args.outdir, "%s_f%d.pt" % (args.tag, fold)))
    all_sp = np.concatenate(all_sp); all_hp = np.concatenate(all_hp)
    all_orig = np.concatenate(all_orig)
    log("-" * 78)
    log("OOF 汇总 (%d 个预测, %.1f 分钟)" % (len(all_sp), (time.time() - t0) / 60))
    log("  soft-argmax:")
    r_soft = C.report_rmse(all_sp, all_orig, all_keys, verbose=True)
    log("  hard-argmax:")
    r_hard = C.report_rmse(all_hp, all_orig, all_keys, verbose=True)
    np.savez(os.path.join(args.outdir, "%s_oof.npz" % args.tag),
             soft=all_sp, hard=all_hp, orig=all_orig, keys=np.array(all_keys))
    log("OOF_RMSE_soft=%.4f  OOF_RMSE_hard=%.4f" % (r_soft, r_hard))
    logf.close()


META = C.load_meta()

if __name__ == "__main__":
    main()