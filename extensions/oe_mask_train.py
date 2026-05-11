#Outlier Exposure a livello di maschera per EoMT.
#Aggiunge la classe 19 come "anomalia" e addestra il modello a rifiutare
#intere maschere corrispondenti a oggetti COCO incollati.

import os
import sys
import math
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image
import numpy as np

# Aggancia i moduli EoMT già presenti nel progetto
project_root = "/content/MLME26"
sys.path.append(project_root)

from eomt.datasets.transforms import Transforms
from eomt.training.mask_classification_semantic import MaskClassificationSemantic

# Costanti
NEW_NUM_CLASSES = 20          # 19 classi Cityscapes originali + classe 19 = anomalia
ANOMALY_CLASS = 19
IGNORE_INDEX = 255


# ----------------------------------------------------------------------
# Dataset – carica immagini e maschere con anomalia = 19
# ----------------------------------------------------------------------
class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(640, 640)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.img_paths = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))
        assert len(self.img_paths) == len(self.mask_paths), (
            f"Numero immagini ({len(self.img_paths)}) ≠ maschere ({len(self.mask_paths)})"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Ridimensioniamo a 640x640 (bilineare per immagine, nearest per maschera)
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        mask_np = np.array(mask).astype(np.int64)

        # Converti la maschera semantica nel formato panottico EoMT
        masks, labels = [], []
        for lbl in np.unique(mask_np):
            if lbl == IGNORE_INDEX:
                continue
            binary = mask_np == lbl
            masks.append(torch.from_numpy(binary))
            labels.append(torch.tensor(lbl, dtype=torch.long))

        # Il target include anche eventuali maschere di anomalia (lbl=19)
        return img_tensor, {"masks": masks, "labels": labels}


# ----------------------------------------------------------------------
# DataModule con le stesse augmentations di EoMT
# ----------------------------------------------------------------------
class AnomalyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        img_size=(640, 640),
        batch_size=4,
        num_workers=4,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
    ):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_mask_dir = train_mask_dir
        self.val_img_dir = val_img_dir
        self.val_mask_dir = val_mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage=None):
        self.train_dataset = AnomalyDataset(
            self.train_img_dir, self.train_mask_dir, self.img_size
        )
        self.val_dataset = AnomalyDataset(
            self.val_img_dir, self.val_mask_dir, self.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._train_collate,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._eval_collate,
        )

    def _train_collate(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        # Applica le trasformazioni EoMT (scaling, jitter) su ogni elemento
        transformed_targets = []
        for img, tgt in zip(imgs, targets):
            tgt["is_crowd"] = [False] * len(tgt["labels"])
            _, masks_t, labels_t = self.transforms(img, tgt["masks"], tgt["labels"])
            transformed_targets.append({"masks": masks_t, "labels": labels_t})
        return imgs, transformed_targets

    def _eval_collate(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, list(targets)


# ----------------------------------------------------------------------
# Costruzione del modello (20 classi, senza testa di classificazione)
# ----------------------------------------------------------------------
def build_model(config_path, ckpt_path, lr=1e-4, load_class_head=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Encoder
    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = enc_cfg["class_path"].rsplit(".", 1)
    EncoderCls = getattr(sys.modules[enc_mod], enc_cls)
    img_size = config.get("data", {}).get("init_args", {}).get("img_size", (640, 640))
    encoder = EncoderCls(img_size=img_size, **enc_cfg.get("init_args", {}))

    # Network
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    NetCls = getattr(sys.modules[net_mod], net_cls)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = NetCls(
        masked_attn_enabled=False,
        num_classes=NEW_NUM_CLASSES,    # 20
        encoder=encoder,
        **net_kwargs,
    )

    # Lightning module
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    LitCls = getattr(sys.modules[lit_mod], lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]
    model_kwargs.update(
        lr=lr,
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=load_class_head,
    )

    model = LitCls(
        img_size=img_size,
        num_classes=NEW_NUM_CLASSES,
        network=network,
        **model_kwargs,
    )

    # Carica i pesi pre‑addestrati saltando la testa di classificazione
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    filtered = {
        k: v
        for k, v in state.items()
        if not any(part in k for part in ["class_head", "class_predictor"])
    }
    # Interpolazione pos_embed se necessario
    pos_key = "network.encoder.backbone.pos_embed"
    if pos_key in filtered:
        ckpt_pos = filtered[pos_key]
        model_pos = model.state_dict()[pos_key]
        if ckpt_pos.shape != model_pos.shape:
            dim = ckpt_pos.shape[-1]
            ckpt_size = int(math.sqrt(ckpt_pos.shape[1]))
            model_size = int(math.sqrt(model_pos.shape[1]))
            ckpt_2d = ckpt_pos.reshape(1, ckpt_size, ckpt_size, dim).permute(0, 3, 1, 2)
            interp = F.interpolate(
                ckpt_2d, size=(model_size, model_size), mode="bicubic", align_corners=False
            )
            filtered[pos_key] = interp.permute(0, 2, 3, 1).reshape(1, model_size * model_size, dim)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Chiavi mancanti (head):", missing)
    print("Chiavi inattese:", unexpected)

    # Opzionale: pesa di più la classe 19 per compensare uno sbilanciamento
    # model.criterion.empty_weight[19] = 5.0
    return model


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/content/MLME26/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    parser.add_argument("--ckpt", default="/content/MLME26/eomt/bin/eomt_cityscapes.bin")
    parser.add_argument("--train_img_dir", required=True)
    parser.add_argument("--train_mask_dir", required=True)
    parser.add_argument("--val_img_dir", required=True)
    parser.add_argument("--val_mask_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir", default="/content/MLME26/checkpoints_oe_mask")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    dm = AnomalyDataModule(
        train_img_dir=args.train_img_dir,
        train_mask_dir=args.train_mask_dir,
        val_img_dir=args.val_img_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = build_model(args.config, args.ckpt, lr=args.lr)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="oe_mask-{epoch:02d}-{val_iou_all:.3f}",
        monitor="val_iou_all",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=args.output_dir, name="oe_mask")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        fast_dev_run=args.debug,
    )

    trainer.fit(model, dm)
    print(f"Training completato. I pesi sono salvati in {args.output_dir}")


if __name__ == "__main__":
    main()