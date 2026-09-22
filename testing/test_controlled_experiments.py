"""Small CPU checks. These never launch the tooth training experiments."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from experiments.controlled.data import (ControlledDataset, EpochSampler, alpha_from_counts,
                                         case_id, evaluation_records, split_files, stage_index)
from experiments.controlled.metrics import Metrics
from experiments.controlled.train import checkpoint_callbacks


def test_split_keeps_paired_jaws_together():
    files = [f'data_{i:03d}_{jaw}.pt' for i in range(100) for jaw in ('upper', 'lower')]
    splits = split_files(files)
    assert {k: len(v) for k, v in splits.items()} == {'train': 140, 'val': 30, 'test': 30}
    ids = [{case_id(f) for f in splits[k]} for k in ('train', 'val', 'test')]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
    assert splits == split_files(list(reversed(files)))


def test_final_growth_stage_reachable_and_no_out_of_bounds():
    rng = np.random.default_rng(5)
    stages = [stage_index(8, 1.5, rng) for _ in range(10000)]
    assert set(stages) == set(range(8))
    assert stage_index(1, 2.5, rng) == 0


def test_softened_weights_reduce_asymmetry_and_reject_unseen_class():
    counts = np.full(17, 1600.)
    counts[1] = 100.
    inverse, softened = alpha_from_counts(counts, 1), alpha_from_counts(counts, .5)
    assert float(inverse[1] / inverse[2]) == pytest.approx(16)
    assert float(softened[1] / softened[2]) == pytest.approx(4)
    assert float(softened.mean()) == pytest.approx(1)
    counts[16] = 0
    with pytest.raises(ValueError):
        alpha_from_counts(counts, 1)


def test_fixed_evaluation_record_pairs_have_same_geometry_seed():
    records = evaluation_records(['data_case_upper.pt'], 42)
    assert len(records) == 10
    for a, b in zip(records[::2], records[1::2]):
        assert (a['file'], a['seed'], a['mode']) == (b['file'], b['seed'], b['mode'])
        assert a['dropout'] == 0 and b['dropout'] == .6


def test_failure_metrics_measure_absent_terminals_and_pair_merges():
    m = Metrics()
    truth = np.array([4]*20 + [5]*20 + [0]*10)
    pred = np.array([4]*40 + [16]*10)
    m.add(truth, pred, np.full(50, .5))
    r = m.result()
    assert r['pair_merge_proxy'] == 1
    assert r['pair_trials'] == 1
    assert r['terminal'][16]['false_positive_mm2'] == 5
    assert r['false_terminal_mm2_per_absent_event'] == 2.5
    assert r['false_identities_per_patch'] == 1


class Toy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.seen = []

    def training_step(self, batch, index):
        self.seen.append(int(batch.item()))
        return (self.weight - 1).square()

    def validation_step(self, batch, index):
        # Best at epoch 0; later epochs deliberately never improve.
        self.log('val_miou', float(1 / (self.current_epoch + 1)))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.1)


class EpochKeys(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, key):
        epoch, index = key
        return epoch * 2 + index


def toy_trainer(path, epochs):
    return pl.Trainer(accelerator='cpu', max_epochs=epochs, logger=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      num_sanity_val_steps=0, callbacks=checkpoint_callbacks(path))


def test_last_checkpoint_and_epoch_keys_survive_resume(tmp_path):
    dataset = EpochKeys()
    train = DataLoader(dataset, sampler=EpochSampler(len(dataset)), num_workers=2,
                       persistent_workers=True)
    val = DataLoader(torch.zeros(1))
    first = Toy()
    toy_trainer(tmp_path, 2).fit(first, train, val)
    last_path = tmp_path / 'last.ckpt'
    last = torch.load(last_path, weights_only=False)
    assert last['epoch'] == 1 and last['global_step'] == 4
    best = torch.load(next(tmp_path.glob('best-*.ckpt')), weights_only=False)
    assert best['epoch'] == 0
    second = Toy()
    resumed_train = DataLoader(dataset, sampler=EpochSampler(len(dataset), start_epoch=2),
                               num_workers=2, persistent_workers=True)
    toy_trainer(tmp_path, 3).fit(second, resumed_train, val, ckpt_path=str(last_path))
    resumed = torch.load(last_path, weights_only=False)
    assert resumed['epoch'] == 2 and resumed['global_step'] == 6
    assert first.seen == [0, 1, 2, 3]
    assert second.seen == [4, 5]


def test_real_patch_reproducible_across_calls_and_color_conditions():
    root = Path('data/3dteethseg/processed_w5')
    files = sorted(root.glob('data_*_upper.pt'))
    if not files:
        pytest.skip('Local mesh cache unavailable')
    filename = files[0].name
    records = evaluation_records([filename], 1042)
    ds = ControlledDataset(root, [filename], records=records, max_faces=256, augment=False)
    a, meta_a, area_a = ds[0]
    b, meta_b, area_b = ds[1]
    repeated, _, _ = ds[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert torch.equal(a[1][:, :24], b[1][:, :24])
    assert not torch.equal(a[1][:, 24:], b[1][:, 24:])
    assert torch.equal(a[1], repeated[1])
    assert torch.equal(area_a, area_b)
    assert meta_a['dropout'] == 0 and meta_b['dropout'] == .6


def first_sample(batch):
    return batch[0]


def test_training_samples_independent_of_worker_count():
    root = Path('data/3dteethseg/processed_w5')
    files = sorted(root.glob('data_*_upper.pt'))
    if not files:
        pytest.skip('Local mesh cache unavailable')
    ds = ControlledDataset(root, [files[0].name], length=2, max_faces=256, targeted=True)
    a = list(DataLoader(ds, sampler=EpochSampler(2, start_epoch=3), num_workers=0,
                        collate_fn=first_sample))
    b = list(DataLoader(ds, sampler=EpochSampler(2, start_epoch=3), num_workers=2,
                        collate_fn=first_sample))
    for (sa, ma, aa), (sb, mb, ab) in zip(a, b):
        assert torch.equal(sa[1], sb[1]) and torch.equal(sa[2], sb[2])
        assert ma == mb and torch.equal(aa, ab)
