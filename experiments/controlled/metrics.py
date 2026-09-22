"""Per-class, upper/lower and failure-focused metrics with explicit denominators."""
from collections import defaultdict
import numpy as np

from experiments.controlled.data import PAIRS


class Metrics:
    def __init__(self):
        self.confusion = np.zeros((17, 17), dtype=np.int64)
        self.patches = self.absent_events = self.pair_trials = self.pair_merges = 0
        self.false_terminal_area = self.absent_patch_area = 0.
        self.false_identity_sum = 0
        self.terminal = {k: {'absent_patches': 0, 'false_positive_mm2': 0.,
                             'false_positive_patches': 0} for k in (1, 16)}

    def add(self, truth, prediction, area):
        truth, prediction, area = map(np.asarray, (truth, prediction, area))
        self.confusion += np.bincount(17 * truth + prediction, minlength=289).reshape(17, 17)
        self.patches += 1
        present = set(truth.tolist())
        self.false_identity_sum += len((set(prediction.tolist()) - present) - {0})
        for k in (1, 16):
            if k not in present:
                fp = float(area[prediction == k].sum())
                self.terminal[k]['absent_patches'] += 1
                self.terminal[k]['false_positive_mm2'] += fp
                self.terminal[k]['false_positive_patches'] += int(fp > 0)
                self.absent_events += 1
                self.false_terminal_area += fp
                self.absent_patch_area += float(area.sum())
        for a, b in PAIRS:
            # Require meaningful visible support; this is a semantic merge proxy,
            # not a connected-instance matching score or guidance-completion score.
            if (truth == a).sum() >= 20 and (truth == b).sum() >= 20:
                modes = [np.bincount(prediction[truth == k], minlength=17).argmax() for k in (a, b)]
                self.pair_trials += 1
                self.pair_merges += int(modes[0] == modes[1] and modes[0] != 0)

    def result(self):
        c = self.confusion
        tp = c.diagonal().astype(float)
        true, predicted = c.sum(1), c.sum(0)
        union = true + predicted - tp
        ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        iou = ratio(tp, union)
        return {'patches': self.patches, 'accuracy': float(tp.sum()/max(1, c.sum())),
                'macro_iou_present_union': float(iou[union > 0].mean()) if np.any(union) else 0.,
                'precision': ratio(tp, predicted).tolist(), 'recall': ratio(tp, true).tolist(),
                'iou': iou.tolist(), 'true_faces': true.tolist(), 'predicted_faces': predicted.tolist(),
                'confusion': c.tolist(), 'false_identities_per_patch': self.false_identity_sum/max(1, self.patches),
                'terminal': self.terminal,
                'false_terminal_mm2_per_absent_event': self.false_terminal_area/max(1, self.absent_events),
                'false_terminal_area_fraction': self.false_terminal_area/max(1e-12, self.absent_patch_area),
                'absent_terminal_events': self.absent_events,
                'pair_merge_proxy': self.pair_merges/max(1, self.pair_trials),
                'pair_trials': self.pair_trials}


class StratifiedMetrics:
    def __init__(self):
        self.groups = defaultdict(Metrics)

    def add(self, truth, pred, area, meta):
        area_bin = 'small' if meta['area_mm2'] < 90 else 'medium' if meta['area_mm2'] < 360 else 'large'
        color = f'dropout={meta["dropout"]}'
        keys = ['all', meta['arch'], 'mode=' + meta['mode'], 'area=' + area_bin,
                color, f'{meta["arch"]}/{meta["mode"]}/{color}']
        present = set(np.asarray(truth).tolist())
        for label, in_arch in zip((1, 16), meta['terminal_in_arch']):
            status = 'absent_from_arch' if not in_arch else 'visible' if label in present else 'not_visible'
            keys.append(f'terminal{label}={status}/{color}')
        for k in keys:
            self.groups[k].add(truth, pred, area)

    def result(self):
        return {k: v.result() for k, v in sorted(self.groups.items())}
