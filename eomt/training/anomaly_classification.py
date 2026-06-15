import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
from torch.optim import AdamW

# New imports for visualization and logging on wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb

class AnomalyClassificationModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ckpt_path: str = None,
        img_size: tuple[int, int] = (1024, 1024), 
        anomaly_class_idx: int = 19, 
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.weight_decay = weight_decay
        self.img_size = img_size
        self.anomaly_class_idx = anomaly_class_idx

        if ckpt_path:
            print(f"Loading base weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            
            ckpt = {k.replace("._orig_mod", ""): v for k, v in ckpt.items()}
            self.load_state_dict(ckpt, strict=False)
            print("Base weights loaded! Initializing anomaly head from scratch.")

        # parameters for semantic segmentation are frozen
        for param in self.network.parameters():
            param.requires_grad = False
        
        # parameters for anomaly segmentation are unfrozen
        for param in self.network.anomaly_head.parameters():
            param.requires_grad = True

        self.save_hyperparameters(ignore=["network", "_class_path"])

    def configure_optimizers(self):
        return AdamW(
            self.network.anomaly_head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


    ################################################################################################
    # WINDOWING HELPER METHODS
    def scale_img_size(self, size: tuple[int, int]):
        factor = max(self.img_size[0] / size[0], self.img_size[1] / size[1])
        return [round(s * factor) for s in size]

    def window_imgs(self, imgs, stride_ratio=0.5):
        """
        Extracts overlapped crops using the sliding window 2D logic
        stride_ratio = 0.5 means that the window advances by half its size (50% overlap).
        """
        crops, origins = [], []
        window_h, window_w = self.img_size
        
        # Compute pixel stride
        stride_h = max(1, int(window_h * stride_ratio))
        stride_w = max(1, int(window_w * stride_ratio))

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size(img.shape[-2:])
            
            resized_img = F.interpolate(
                img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)

            # Generate the starting coordinates for H and W
            y_starts = list(range(0, max(1, new_h - window_h + 1), stride_h))
            # Ensure the last crop reaches exactly the bottom edge
            if new_h > window_h and y_starts[-1] + window_h < new_h:
                y_starts.append(new_h - window_h)

            x_starts = list(range(0, max(1, new_w - window_w + 1), stride_w))
            # Ensure the last crop reaches exactly the right edge
            if new_w > window_w and x_starts[-1] + window_w < new_w:
                x_starts.append(new_w - window_w)

            # Extract crops by sliding across the 2D grid
            for y in y_starts:
                for x in x_starts:
                    crop = resized_img[:, y:y+window_h, x:x+window_w]
                    
                    # If the original image is smaller than the window, apply padding
                    actual_h, actual_w = crop.shape[-2:]
                    if actual_h < window_h or actual_w < window_w:
                        pad_b = window_h - actual_h
                        pad_r = window_w - actual_w
                        crop = F.pad(crop, (0, pad_r, 0, pad_b), mode='reflect')

                    crops.append(crop)
                    
                    # Save the origin in a 2D format: (index_img, y_start, y_end, x_start, x_end)
                    origins.append((i, y, y+actual_h, x, x+actual_w))

        return torch.stack(crops), origins

    def revert_window_logits(self, crop_logits, origins, img_sizes):
        """
        Ricostruisce l'immagine intera facendo la media (sums / counts) 
        nelle zone di sovrapposizione dei crop 2D.
        """
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size(size)
            logit_sums.append(torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device))
            logit_counts.append(torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device))

        for crop_i, (img_i, y1, y2, x1, x2) in enumerate(origins):

            # Compute the valid dimensions of the crop (excluding padding, if present)
            h_valid = y2 - y1
            w_valid = x2 - x1

            # Sum the logits and increase the counter for the mean
            logit_sums[img_i][:, y1:y2, x1:x2] += crop_logits[crop_i, :, :h_valid, :w_valid]
            logit_counts[img_i][:, y1:y2, x1:x2] += 1

        # Prevent division by zero in rare cases of border artifacts
        for counts in logit_counts:
            counts[counts == 0] = 1

        return [
            F.interpolate((sums / counts)[None, ...], img_sizes[i], mode="bilinear", align_corners=False)[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]
    def _get_anomaly_masks_from_targets(self, targets, img_sizes, device):
        B = len(targets)
        anomaly_masks = []
        for i in range(B):
            target = targets[i]
            H, W = img_sizes[i]
            mask = torch.zeros((1, H, W), device=device)
            for j, label in enumerate(target["labels"]):
                if label == self.anomaly_class_idx:
                    mask[0] = torch.max(mask[0], target["masks"][j].float())
            anomaly_masks.append(mask)
        return torch.stack(anomaly_masks)

    @torch.no_grad()
    def _log_training_image(self, img, gt_mask, pred_logit):
        """
        Genera un plot a 3 pannelli (Immagine, Ground Truth, Predizione) 
        e lo carica su Weights & Biases.
        """
        # Convert the image in HWC uint8 (0-255) format
        img_np = img.clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype('uint8')
        gt_np = gt_mask.squeeze(0).cpu().numpy()
        # Apply sigmaoid to obtain probabilities [0,1] 
        pred_np = torch.sigmoid(pred_logit).squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Anomaly")
        axes[1].axis("off")

        im = axes[2].imshow(pred_np, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Predicted Probability")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        
        # Save the plot in the memory as PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf)
        
        # Log to wandb via the active experiment in the Lightning logger
        if self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                "train/predictions_visual": wandb.Image(
                    pil_img, 
                    caption=f"Epoch {self.current_epoch} - Train Batch 0"
                )
            })

    
    #### TRAINING AND VALIDATION
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        if isinstance(imgs, (list, tuple)):
            imgs = torch.stack(imgs)
        imgs = imgs.float()
        img_sizes = [img.shape[-2:] for img in imgs]
        
        anomaly_masks = self._get_anomaly_masks_from_targets(targets, img_sizes, imgs.device)
        crops, origins = self.window_imgs(imgs)
        
        x = crops / 255.0
        _, _, anomaly_logits = self.network(x)
        
        crop_logits = F.interpolate(anomaly_logits[-1], size=self.img_size, mode="bilinear", align_corners=False)
        
        reverted_logits = self.revert_window_logits(crop_logits, origins, img_sizes)
        reverted_logits = torch.stack(reverted_logits)
        
        loss = F.binary_cross_entropy_with_logits(reverted_logits, anomaly_masks)
        
        #LOGGING ON WANDB
        # If it is the first step of the current epoch and the logger is configured, log the plot
        if batch_idx == 0:
            try:
                # Use .detach() to avoid computational graph issues
                self._log_training_image(imgs[0].detach(), anomaly_masks[0].detach(), reverted_logits[0].detach())
            except Exception as e:
                print(f"Errore durante il logging dell'immagine su wandb: {e}")
        ##############################################################################
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        if isinstance(imgs, (list, tuple)):
            imgs = torch.stack(imgs)
        imgs = imgs.float()
        img_sizes = [img.shape[-2:] for img in imgs]
        
        anomaly_masks = self._get_anomaly_masks_from_targets(targets, img_sizes, imgs.device)
        crops, origins = self.window_imgs(imgs)
        
        x = crops / 255.0
        _, _, anomaly_logits = self.network(x)
        
        crop_logits = F.interpolate(anomaly_logits[-1], size=self.img_size, mode="bilinear", align_corners=False)
        
        reverted_logits = self.revert_window_logits(crop_logits, origins, img_sizes)
        reverted_logits = torch.stack(reverted_logits)
        
        loss = F.binary_cross_entropy_with_logits(reverted_logits, anomaly_masks)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss