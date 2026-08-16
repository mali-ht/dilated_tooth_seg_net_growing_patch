import lightning as L
import torch
import torchmetrics as tm

from dataset.patch_losses import FocalLoss
from models.patch_dilated_tooth_seg_network import PatchDilatedToothSegmentationNetwork

# Component 5 (claude_code_prompt.md) - adapted from models/dilated_tooth_seg_network.py's
# LitDilatedToothSegmentationNetwork (untouched). Same overall shape (Adam + StepLR, matching
# the original paper's "Experiment Setup" hyperparameters), but:
#   - wraps PatchDilatedToothSegmentationNetwork (Component 4) instead of the original network
#   - training_step/validation_step consume a PatchBatch (models/patch_collate.py) instead of a
#     plain (pos, x, y) tuple - it already carries area_mm2 and the precomputed neighbor indices
#     the model's forward() needs
#   - loss is FocalLoss (dataset/patch_losses.py), not plain CrossEntropyLoss - TRAINING_CONCERNS.md
#     #5's confirmed must-do for the skewed gum/tooth ratio in small patches
#   - mIoU uses torchmetrics.JaccardIndex(average="macro") directly: verified empirically (this
#     session) that torchmetrics 1.9.0 already excludes classes with zero union from the macro
#     average while still penalizing false-positive-only classes - exactly Component 5's
#     "patch-aware — ignore classes absent from a patch" ask, natively, so no custom metric class
#     was needed
#   - num_classes and dilation_gating are the two real experiment-matrix flags from
#     claude_code_prompt.md Component 5 ("norm scheme" isn't exposed as a flag - GroupNorm is the
#     settled design here, not an open ablation, per the reasoning in models/patch_layer.py)


class PatchLitDilatedToothSegmentationNetwork(L.LightningModule):
    def __init__(self, num_classes: int = 17, feature_dim: int = 24, k: int = 32,
                 dilation_ks=(200, 900, 1800), area_thresholds=(40.0, 180.0, 360.0),
                 dilation_gating: bool = True, focal_gamma: float = 2.0, class_alpha=None,
                 lr: float = 1e-3, weight_decay: float = 1e-5, lr_step_size: int = 60,
                 lr_gamma: float = 0.5):
        super().__init__()
        self.model = PatchDilatedToothSegmentationNetwork(
            num_classes=num_classes, feature_dim=feature_dim, k=k, dilation_ks=dilation_ks,
            area_thresholds=area_thresholds, dilation_gating=dilation_gating)
        self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_alpha)
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        self.save_hyperparameters()

    def _forward_and_loss(self, batch):
        pred = self.model(batch.x, batch.pos, area_mm2=batch.area_mm2,
                           local_idx=batch.local_idx, dilated_idx=batch.dilated_idx)
        pred = pred.transpose(2, 1)  # (B, N, C) -> (B, C, N): FocalLoss/torchmetrics convention
        loss = self.loss_fn(pred, batch.labels)
        return pred, loss

    def training_step(self, batch, batch_idx):
        pred, loss = self._forward_and_loss(batch)
        self.train_acc(pred, batch.labels)
        self.train_miou(pred, batch.labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        # patch area isn't a model metric, but tracking it confirms training is actually seeing
        # the intended broad size distribution (early_bias_power's small/large mix), not some
        # accidental sampling skew - cheap to log, easy to spot a problem from
        self.log("train_area_mm2", batch.area_mm2, prog_bar=False, on_step=True, on_epoch=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self._forward_and_loss(batch)
        self.val_acc(pred, batch.labels)
        self.val_miou(pred, batch.labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def test_step(self, batch, batch_idx):
        pred, loss = self._forward_and_loss(batch)
        self.val_acc(pred, batch.labels)
        self.val_miou(pred, batch.labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

    def predict_labels(self, batch):
        """Real-inference path: no ground truth required (unlike training/validation_step, which
        need batch.labels for the loss/metrics) - just the per-face predicted class ids."""
        with torch.no_grad():
            pred = self.model(batch.x, batch.pos, area_mm2=batch.area_mm2,
                               local_idx=batch.local_idx, dilated_idx=batch.dilated_idx)
            return pred.argmax(dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                      weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }
