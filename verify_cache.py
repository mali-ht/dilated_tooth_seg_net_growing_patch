#!/usr/bin/env python3
"""Verify the processed_w5 cache: split coverage + deep per-file integrity.

Written 2026-08-22 after a partial transfer to the Blackwell server left 374 of 1800 cases
missing and one file (data_0IU0UV8E_upper.pt) written with a valid pickle header but 75.8%
null bytes. The dataset class filters the split list through os.path.isfile, so missing files
are skipped SILENTLY; the corrupt one only surfaced as an UnpicklingError in a DataLoader
worker at the end of epoch 0, after ~5 minutes of training per lane.

Run this after any data transfer, before launching a long campaign:
    python verify_cache.py
"""
import glob, os, pickle, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT, PROC = "data/3dteethseg", "processed_w5"

def check(p):
    try:
        with open(p, "rb") as f:
            fa, tri, vn, fn, lab = pickle.load(f)
        fa, tri, lab, vn, fn = fa.numpy(), tri.numpy(), lab.numpy(), vn.numpy(), fn.numpy()
        e = []
        if fa.ndim != 2 or fa.shape[1] != 3:   e.append(f"faces shape {fa.shape}")
        if tri.shape[0] != fa.shape[0]:        e.append("triangles/faces length mismatch")
        if lab.shape[0] != fa.shape[0]:        e.append("labels/faces length mismatch")
        if fa.min() < 0:                       e.append("negative face index")
        if not (np.isfinite(tri).all() and np.isfinite(fn).all() and np.isfinite(vn).all()):
            e.append("non-finite values")
        u = np.unique(lab)
        if u.min() < 0 or u.max() > 16:        e.append(f"labels out of range [{u.min()},{u.max()}]")
        ext = tri.reshape(-1, 3).ptp(axis=0)
        if not (5 < ext.max() < 500):          e.append(f"implausible extent {ext.round(1).tolist()}")
        return (os.path.basename(p), e) if e else None
    except Exception as ex:
        return (os.path.basename(p), [f"{type(ex).__name__}: {ex}"])

def main():
    files = sorted(glob.glob(f"{ROOT}/{PROC}/*.pt"))
    if not files: sys.exit(f"no .pt files under {ROOT}/{PROC}")
    L = lambda n: {l.strip() for l in open(f"{ROOT}/raw/{n}.txt") if l.strip()}
    try:
        tr = L("training_lower") | L("training_upper")
        te = L("testing_lower") | L("testing_upper")
    except FileNotFoundError as e:
        sys.exit(f"missing split file: {e}. Get them from the Teeth3DS_split/ dir of "
                 "github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges")
    cached = {os.path.basename(p)[5:-3] for p in files}
    print(f"split coverage:  train {len(cached&tr)}/{len(tr)}   val {len(cached&te)}/{len(te)}")
    missing = (tr | te) - cached
    if missing:
        print(f"  WARNING: {len(missing)} case(s) in the split lists are NOT cached and will be")
        print(f"           SILENTLY SKIPPED, shrinking the dataset without any error.")

    bad = []
    with ProcessPoolExecutor(max_workers=min(64, os.cpu_count() or 8)) as ex:
        for r in ex.map(check, files, chunksize=4):
            if r: bad.append(r)
    print(f"integrity:       {len(files)-len(bad)}/{len(files)} files OK")
    for n, e in bad:
        print(f"  CORRUPT {n}: {'; '.join(e)}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
