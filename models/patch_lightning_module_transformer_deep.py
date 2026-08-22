import lightning as L
import torch
import torchmetrics as tm

from dataset.patch_losses import FocalLoss
from models.patch_dilated_tooth_seg_transformer_deep_network import PatchDilatedToothSegTransformerDeepNetwork

# New, additive-only file - models/patch_lightning_module.py and
# models/patch_lightning_module_transformer.py are both untouched. Same training/validation/
# optimizer setup, wrapping PatchDilatedToothSegTransformerDeepNetwork
# (models/patch_dilated_tooth_seg_transformer_deep_network.py) - duplicated rather than
# subclassed, since __init__ constructs self.model inline with no factory hook to override
# cleanly (same "duplicate __init__" precedent as elsewhere in this project).


class PatchLitDilatedToothSegTransformerDeepNetwork(L.LightningModule):
    def __init__(self, num_classes: int = 17, feature_dim: int = 24, k: int = 32,
                 dilation_ks=(200, 900, 1800), area_thresholds=(40.0, 180.0, 360.0),
                 dilation_gating: bool = True, transformer_tokens: int = 256,
                 transformer_layers: int = 2, transformer_heads: int = 4,
                 late_transformer_embed_dim: int = 128,
                 focal_gamma: float = 2.0, class_alpha=None,
                 lr: float = 1e-3, weight_decay: float = 1e-5, lr_step_size: int = 60,
                 lr_gamma: float = 0.5):
        super().__init__()
        self.model = PatchDilatedToothSegTransformerDeepNetwork(
            num_classes=num_classes, feature_dim=feature_dim, k=k, dilation_ks=dilation_ks,
            area_thresholds=area_thresholds, dilation_gating=dilation_gating,
            transformer_tokens=transformer_tokens, transformer_layers=transformer_layers,
            transformer_heads=transformer_heads, late_transformer_embed_dim=late_transformer_embed_dim)
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
