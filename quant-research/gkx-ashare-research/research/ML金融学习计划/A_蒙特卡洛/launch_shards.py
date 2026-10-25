"""
并行跑多个分片：把 100 次重复按种子切成若干段，各起一个子进程同时跑。

后台任务数有上限（10 个），但一个任务里可以起任意多个子进程，所以这里把
12 个分片塞进一个任务。子进程 stdout 直接继承，输出会混在同一个日志里，
每行带 tag 前缀不好分辨时，看 "[rep k/n] seed=..." 的 seed 就知道是哪一段。

用法：
    python launch_shards.py tree --pc 50 --model a --shards 12
    python launch_shards.py tree --pc 50 --model a --shards 12 --dry
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def split_reps(total: int, shards: int) -> list[tuple[int, int]]:
    """把 total 次重复尽量均匀地切成 shards 段，返回 [(seed0, reps), ...]。"""
    base, extra = divmod(total, shards)
    out, seed = [], 0
    for i in range(shards):
        reps = base + (1 if i < extra else 0)
        out.append((seed, reps))
        seed += reps
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("tree", "nn", "lin"), help="跑哪一类模型")
    ap.add_argument("--pc", type=int, default=50, choices=(50, 100))
    ap.add_argument("--model", default="a", choices=("a", "b"))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rf-trees", default="300")
    ap.add_argument("--gb-trees", default="300")
    ap.add_argument("--rf-depth", default="1,2,3,4,5,6")
    ap.add_argument("--rf-mfeat", default="3,5,10,20,30,50")
    ap.add_argument("--gb-lr", default="0.001,0.01")
    ap.add_argument("--nnes", default="10")
    ap.add_argument("--nep", default="100")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="默认给每个分片加 --resume；加这个开关则从头跑")
    args = ap.parse_args()

    tag = args.tag or f"A1_{args.kind.upper()}_pc{args.pc}_{args.model}"
    parts = split_reps(args.reps, args.shards)

    common = ["-u", "run_mc.py", "--pc", str(args.pc), "--model", args.model,
              "--models", args.kind, "--theta-scale", "1.08"]
    if args.kind == "tree":
        common += ["--rf-trees", args.rf_trees, "--gb-trees", args.gb_trees,
                   "--rf-depth", args.rf_depth, "--rf-mfeat", args.rf_mfeat,
                   "--gb-lr", args.gb_lr]
    elif args.kind == "nn":
        common += ["--nnes", args.nnes, "--nep", args.nep]

    procs = []
    for i, (seed0, reps) in enumerate(parts):
        cmd = [sys.executable] + common + ["--reps", str(reps),
                                           "--seed0", str(seed0),
                                           "--tag", f"{tag}_s{i}"]
        if not args.no_resume:
            cmd.append("--resume")
        print(f"  分片 {i:2d}: 种子 {seed0}–{seed0 + reps - 1}（{reps} 次）", flush=True)
        if not args.dry:
            procs.append(subprocess.Popen(cmd))
    if args.dry:
        print(f"\n共 {len(parts)} 个分片，覆盖 {args.reps} 次重复")
        return 0

    failed = 0
    for i, p in enumerate(procs):
        if p.wait() != 0:
            failed += 1
            print(f"  分片 {i} 退出码 {p.returncode}", flush=True)
    print(f"全部分片结束，失败 {failed} 个。合并命令：python merge_raw.py {tag}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
