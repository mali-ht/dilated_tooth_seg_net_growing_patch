#!/bin/bash
# At-a-glance status of the experiment lanes. Read-only; safe to run during training.
# NB: val_miou is read from CHECKPOINT FILENAMES, not the progress bar. `tmux new -d` creates an
# 80-col pane, so tqdm truncates its postfix mid-metric ("val_loss=0.402, va") and the value is
# simply absent from the log. ModelCheckpoint writes best-epoch=N-val_miou=X.ckpt, which is both
# untruncated and the number you actually care about.
cd "$(dirname "${BASH_SOURCE[0]}")"
CK=checkpoints/patch_dilated_tooth_seg_net
printf "%-46s %-7s %-15s %-9s %s\n" EXPERIMENT EPOCH TRAIN BEST_MIOU "AT_EPOCH"
printf '%.0s-' {1..95}; echo
for f in $(ls -t logs/experiments/*.log 2>/dev/null | head -6); do
  python3 - "$f" "$CK" <<'PY'
import re, sys, os, glob
f, ck = sys.argv[1], sys.argv[2]
d = re.sub(r'\x1b\[[0-9;]*m','',open(f,'rb').read().decode('utf8','replace').replace('\r','\n'))
ep = (re.findall(r'Epoch (\d+):', d) or ['-'])[-1]
pr = (re.findall(r'Epoch \d+:\s*(\d+%)\|[^|]*\|\s*(\d+/\d+)', d) or [('-','')])[-1]
ver = (re.findall(r"experiment_version='([^']+)'", d) or [None])[0]
best, bep = '-', '-'
if ver:
    cks = glob.glob(os.path.join(ck, ver, 'best-epoch=*.ckpt'))
    if cks:
        m = re.search(r'best-epoch=(\d+)-val_miou=([\d.]+?)\.ckpt', os.path.basename(cks[0]))
        if m: bep, best = m.group(1), m.group(2)
name = (ver or os.path.basename(f)).replace('Patch_Seg_17class_','')
name = name if len(name)<=46 else '...'+name[-43:]
print("%-46s %-7s %-15s %-9s %s" % (name, ep, f"{pr[0]} {pr[1]}", best, bep))
PY
done
