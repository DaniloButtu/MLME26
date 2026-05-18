# Copyright (c) OpenMMLab. All rights reserved.
# Anomaly detection evaluation script for EoMT.
#
# Supports two operating modes, selected automatically based on the config:
#
#   1. AnomalyClassificationModule  (eomt_mlp.yaml)
#      The network has anomaly_head_enabled=True.
#      Produces: MLP anomaly map  +  classic segmentation score.
#      Final score = max(mlp_map, classic_map_normalised).
#
#   2. MaskClassificationSemantic   (eomt_base_640.yaml, or any standard config)
#      The network has no anomaly_head.
#      Produces: classic segmentation score only.
#      Final score = classic_map_normalised.
#
# Usage examples:
#   # Anomaly model, FS LostFound
#   python eval/evalAnomaly_eomt.py \
#       --config_path eomt/configs/dinov2/cityscapes/semantic/eomt_mlp.yaml \
#       --ckpt_path   eomt/hdacek0x/checkpoints/epoch=49-step=5000.ckpt \
#       --input       "dataset/FS_LostFound_full/images/*.*" \
#       --method      msp
#
#   # Baseline model, RoadAnomaly21
#   python eval/evalAnomaly_eomt.py \
#       --config_path eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
#       --ckpt_path   eomt/bin/eomt_cityscapes.bin \
#       --input       "dataset/RoadAnomaly21/images/*.*" \
#       --method      max_entropy

import os
import re
import sys
import cv2
import glob
import torch
import random
import yaml
import importlib
import warnings
import math
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from lightning import seed_everything

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
eomt_path = os.path.join(project_root, 'eomt')
if eomt_path not in sys.path:
    sys.path.append(eomt_path)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed = 42
seed_everything(seed, verbose=False)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES  = 19

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_pos_embed(state_dict, model, encoder_cfg, img_size):
    """Bicubic-interpolate pos_embed when the checkpoint resolution differs."""
    key = 'network.encoder.backbone.pos_embed'
    if key not in state_dict:
        return
    ckpt_pe  = state_dict[key]
    model_pe = model.state_dict().get(key)
    if model_pe is None or ckpt_pe.shape == model_pe.shape:
        return

    print(f"Interpolating pos_embed: {ckpt_pe.shape} -> {model_pe.shape}")
    dim        = ckpt_pe.shape[-1]
    patch_size = encoder_cfg.get("init_args", {}).get("patch_size", 16)
    target_h   = img_size[0] // patch_size
    target_w   = img_size[1] // patch_size
    ckpt_seq   = ckpt_pe.shape[1]
    ckpt_h     = int(math.sqrt(ckpt_seq))  # assumes square; works for 1024x1024 source
    ckpt_w     = ckpt_seq // ckpt_h

    pe_2d        = ckpt_pe.reshape(1, ckpt_h, ckpt_w, dim).permute(0, 3, 1, 2)
    interpolated = F.interpolate(pe_2d, size=(target_h, target_w),
                                 mode='bicubic', align_corners=False)
    state_dict[key] = interpolated.permute(0, 2, 3, 1).reshape(1, target_h * target_w, dim)


def _build_classic_map(mask_logits_per_layer, class_logits_per_layer,
                       revert_fn, origins, img_sizes, img_size, method):
    """
    Reconstruct a per-pixel anomaly score from the segmentation heads.
    Returns a normalised numpy array in [0, 1].
    """
    # Take the final decoder layer output (only layer when masked attn is off)
    mask_logits  = mask_logits_per_layer[-1].float()   # [B, Q, H_tok, W_tok]
    class_logits = class_logits_per_layer[-1].float()  # [B, Q, C+1]

    # Resize mask logits to crop resolution then assemble per-pixel class scores
    # using the official EoMT formula: sigmoid(mask) · softmax(class)[..., :-1]
    mask_logits = F.interpolate(mask_logits, size=img_size,
                                mode="bilinear", align_corners=False)
    # Assemble per-pixel class scores using the official EoMT formula:
    #   crop_logits[b, c, h, w] = Σ_q sigmoid(mask[b,q,h,w]) * softmax(class[b,q])[c]
    # Void class is excluded (softmax[..., :-1]) — identical to to_per_pixel_logits_semantic.
    # We inline it here so this function works with both AnomalyClassificationModule
    # (which has no such method) and MaskClassificationSemantic.
    crop_logits = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )

    # Stitch crops back to full-image resolution
    logits_list = revert_fn(crop_logits, origins, img_sizes)
    logits      = logits_list[0].unsqueeze(0)  # [1, C, H, W]

    # Decide if values are already probabilities or raw logits
    sums = logits.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-2) or logits.min() < 0.0:
        probs = torch.softmax(logits, dim=1)
    else:
        probs = logits

    void_probs    = probs[:, -1, :, :].squeeze(0).cpu().numpy()
    logits_no_void = logits[:, :-1, :, :]

    if method == "msp":
        cmap = 1.0 - torch.max(probs, dim=1)[0].squeeze().cpu().numpy()
    elif method == "max_logit":
        cmap = -np.max(logits_no_void.squeeze(0).cpu().numpy(), axis=0)
    elif method == "max_entropy":
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        cmap = entropy.squeeze(0).cpu().numpy()
    elif method == "rba":
        # Original formula: higher in-distribution confidence → lower anomaly score
        tanh_sum = np.sum(np.tanh(logits_no_void.squeeze(0).cpu().numpy()), axis=0)
        cmap = -tanh_sum
    else:
        raise ValueError(f"Unknown method: {method}")

    cmap = np.nan_to_num(cmap, nan=0.0, posinf=1.0, neginf=0.0)
    # Return raw scores — do NOT normalize here.
    # average_precision_score is computed across all pixels of all images,
    # so per-image min-max normalization would destroy cross-image ranking.
    return cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description="Anomaly detection evaluation for EoMT models."
    )
    parser.add_argument(
        "--input",
        default=["dataset/RoadAnomaly21/images/*.*"],
        nargs="+",
        help="Glob pattern(s) for input images.",
    )
    parser.add_argument(
        "--config_path",
        default="./eomt/configs/dinov2/cityscapes/semantic/eomt_mlp.yaml",
        help="Path to the YAML config that defines model + data.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Path to the checkpoint to evaluate. "
             "If omitted, the script uses ckpt_path from config or no pretrained weights.",
    )
    parser.add_argument('--subset',      default="val")
    parser.add_argument('--num-workers',  type=int, default=4)
    parser.add_argument('--batch-size',   type=int, default=1)
    parser.add_argument('--method', default='rba',
                        choices=['msp', 'max_logit', 'max_entropy', 'rba'])
    parser.add_argument(
        '--img_size', type=int, nargs=2, default=None, metavar=('H', 'W'),
        help="Override the inference crop size (H W). "
             "Defaults to the size in the encoder config (usually 1024 1024). "
             "Use '640 640' for eomt_base_640.yaml to match its training resolution."
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--combine_score', default='max', choices=['raw','max', 'mean','dot','weighted_sum'])
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list       = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    results_file = open('results.txt', 'a')

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module`.*",
    )

    # ------------------------------------------------------------------
    # Detect operating mode from config class_path
    # ------------------------------------------------------------------
    lit_class_path = config["model"]["class_path"]
    # True  -> AnomalyClassificationModule  (eomt_mlp.yaml)
    # False -> MaskClassificationSemantic   (eomt_base_640.yaml)
    is_anomaly_module = "AnomalyClassificationModule" in lit_class_path
    print(f"Mode: {'AnomalyClassificationModule' if is_anomaly_module else 'MaskClassificationSemantic'}")

    # ------------------------------------------------------------------
    # Build encoder
    # ------------------------------------------------------------------
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

    # Use the encoder's own img_size from the model config (NOT the data img_size).
    # The data img_size (e.g. 1024×2048 for training) differs from the encoder's
    # training resolution and would cause a pos_embed shape mismatch.
    enc_kwargs = encoder_cfg.get("init_args", {}).copy()
    img_size   = enc_kwargs.get(
        "img_size",
        config.get("model", {}).get("init_args", {}).get("img_size", [1024, 1024])
    )
    if isinstance(img_size, list):
        img_size = tuple(img_size)

    # --img_size CLI flag overrides whatever the config says.
    # For eomt_base_640.yaml (no img_size in config, defaults to 1024x1024)
    # you should pass --img_size 640 640 to match the model's training resolution.
    if args.img_size is not None:
        img_size = tuple(args.img_size)
        print(f"img_size overridden by --img_size: {img_size}")
    else:
        print(f"img_size from config: {img_size}")

    # Always inject img_size into enc_kwargs — eomt_base_640.yaml omits it
    # from the encoder's init_args (ViT requires it as a positional argument).
    enc_kwargs["img_size"] = list(img_size)

    encoder = encoder_cls(**enc_kwargs)

    # ------------------------------------------------------------------
    # Build network (EoMT)
    # ------------------------------------------------------------------
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)

    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    # Disable masked attention during inference for speed
    network_kwargs["masked_attn_enabled"] = False
    network_kwargs["num_classes"]         = NUM_CLASSES

    network = network_cls(encoder=encoder, **network_kwargs)

    # ------------------------------------------------------------------
    # Build Lightning module
    # ------------------------------------------------------------------
    lit_module_name, lit_class_name = lit_class_path.rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)

    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]
    model_kwargs["img_size"] = img_size

    if is_anomaly_module:
        # AnomalyClassificationModule has no num_classes parameter — remove it if present.
        model_kwargs.pop("num_classes", None)
    else:
        # MaskClassificationSemantic requires num_classes.
        # eomt_base_640.yaml is minimal and omits it, so we supply the default.
        model_kwargs.setdefault("num_classes", NUM_CLASSES)

    # Suppress in-__init__ base-weight loading; we handle checkpoint loading below.
    model_kwargs["ckpt_path"] = None

    model = lit_cls(network=network, **model_kwargs).eval().to(device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    # Priority: --ckpt_path CLI arg > ckpt_path in config root > no checkpoint
    ckpt_path = args.ckpt_path or config.get("ckpt_path") or None

    if ckpt_path:
        print(f"Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove torch.compile artefacts (e.g. "._orig_mod" prefixes)
        state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}

        # Bicubic-interpolate pos_embed when checkpoint and model resolutions differ
        _interpolate_pos_embed(state_dict, model, encoder_cfg, img_size)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")

        if is_anomaly_module:
            head_missing = [k for k in missing if "anomaly_head" in k]
            if head_missing:
                print(f"WARNING: anomaly_head weights not loaded: {head_missing}")
            else:
                print("anomaly_head weights loaded successfully.")
    else:
        print("No checkpoint specified; using randomly initialised weights.")

    # ------------------------------------------------------------------
    # Inference & evaluation loop
    # ------------------------------------------------------------------
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(f"\nProcessing: {path}")

        img_pil    = Image.open(path).convert('RGB')
        img_np     = np.array(img_pil)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if not args.cpu else "cpu"):
            # window_imgs_semantic (MaskClassificationSemantic) converts via PIL internally
            # and therefore requires uint8 tensors.
            # window_imgs (AnomalyClassificationModule) expects float32 in [0, 255].
            if is_anomaly_module:
                imgs_list = [img_tensor.float()]
            else:
                imgs_list = [img_tensor]  # keep uint8 for window_imgs_semantic

            img_sizes = [img.shape[-2:] for img in imgs_list]

            # Windowing: AnomalyClassificationModule uses 2-D sliding window;
            # MaskClassificationSemantic uses 1-D strip windowing.
            mod = model.module if isinstance(model, torch.nn.DataParallel) else model
            if is_anomaly_module:
                crops, origins = mod.window_imgs(imgs_list)
            else:
                crops, origins = mod.window_imgs_semantic(imgs_list)

            # Normalise to [0, 1] before the forward pass
            x = crops.float() / 255.0

            # Single forward pass — all three output lists are always returned.
            # anomaly_logits_per_layer[i] is None when anomaly_head_enabled=False.
            mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = model.network(x)

            # ── MLP HEAD anomaly map (only when head exists) ──────────────
            has_anomaly_head = (
                is_anomaly_module
                and anomaly_logits_per_layer[-1] is not None
            )

            if has_anomaly_head:
                crop_anomaly = anomaly_logits_per_layer[-1]
                crop_anomaly = F.interpolate(crop_anomaly, size=img_size,
                                             mode="bilinear", align_corners=False)
                anomaly_list     = mod.revert_window_logits(crop_anomaly, origins, img_sizes)
                mlp_anomaly_map  = torch.sigmoid(anomaly_list[0]).squeeze(0).cpu().numpy()
                mlp_anomaly_map  = np.nan_to_num(mlp_anomaly_map,
                                                  nan=0.0, posinf=1.0, neginf=0.0)

            # ── Classic segmentation-based score ──────────────────────────
            # The two modules use different windowing strategies with different
            # origin formats, so we pick the matching revert helper:
            #   - AnomalyClassificationModule  -> 2-D origins -> revert_window_logits
            #   - MaskClassificationSemantic   -> 1-D origins -> revert_window_logits_semantic
            if is_anomaly_module:
                revert_fn = mod.revert_window_logits
            else:
                revert_fn = mod.revert_window_logits_semantic

            classic_map = _build_classic_map(
                mask_logits_per_layer, class_logits_per_layer,
                revert_fn, origins, img_sizes, img_size, args.method
            )

            # ── Combine scores ────────────────────────────────────────────
            if has_anomaly_head:
                # Normalize classic map to [0,1] only here, so it matches the
                # MLP sigmoid scale before taking the element-wise maximum.
                c_min, c_max = classic_map.min(), classic_map.max()
                classic_map_norm = (
                    (classic_map - c_min) / (c_max - c_min)
                    if c_max > c_min else np.zeros_like(classic_map)
                )
                if args.combine_score == "max":
                    anomaly_map = np.maximum(mlp_anomaly_map, classic_map_norm)
                elif args.combine_score == "mean":
                    anomaly_map = (mlp_anomaly_map + classic_map_norm) / 2
                elif args.combine_score == "dot":
                    anomaly_map = mlp_anomaly_map * classic_map_norm
                elif args.combine_score == "weighted_sum":
                    anomaly_map = 0.7 * mlp_anomaly_map + 0.3 * classic_map_norm
                elif args.combine_score == "raw":
                    anomaly_map = mlp_anomaly_map
            else:
                # Baseline: use raw classic scores — identical to the original repo.
                anomaly_map = classic_map

            anomaly_result = anomaly_map

            # Save a debug heatmap for the very first image processed
            if len(ood_gts_list) <= 0:
                debug_stem = os.path.splitext(os.path.basename(path))[0]
                map_u8     = cv2.normalize(anomaly_map, None, 0, 255,
                                           cv2.NORM_MINMAX, cv2.CV_8U)
                heatmap    = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)
                debug_name = f"debug_heatmap_{debug_stem}.jpg"
                cv2.imwrite(debug_name, heatmap)
                print(f"\n--- Debug heatmap saved as {debug_name} ---\n")

        # ------------------------------------------------------------------
        # Ground-truth loading and label remapping
        # ------------------------------------------------------------------
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        try:
            mask = Image.open(pathGT)
        except Exception:
            print(f"Skipping {path}: ground truth not found.")
            continue

        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where(ood_gts == 0,  255, ood_gts)
            ood_gts = np.where(ood_gts == 1,  0,   ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where(ood_gts == 14, 255, ood_gts)
            ood_gts = np.where(ood_gts < 20,  0,   ood_gts)
            ood_gts = np.where(ood_gts == 255, 1,   ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        del anomaly_result, ood_gts, mask, img_tensor, img_pil, img_np
        torch.cuda.empty_cache()

    results_file.write("\n")

    if not ood_gts_list:
        print("No valid ground truths found.")
        results_file.close()
        return

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    ood_gts        = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    valid_mask = (ood_gts == 0) | (ood_gts == 1)
    val_out    = anomaly_scores[valid_mask]
    val_label  = ood_gts[valid_mask]

    prc_auc = average_precision_score(val_label, val_out)
    fpr     = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0:.4f}')
    print(f'FPR@TPR95:   {fpr   * 100.0:.4f}')


    input_parts  = re.split(r'[\\/]', str(args.input[0]))
    try:
        ds_idx      = [p.lower() for p in input_parts].index('dataset')
        folder_name = input_parts[ds_idx + 1] if ds_idx + 1 < len(input_parts) else input_parts[0]
    except ValueError:
        folder_name = input_parts[0]

    caps_digits = "".join(re.findall(r'[A-Z0-9]', folder_name))
    if caps_digits:
        dataset_tag = caps_digits                                      # e.g. RA21, RO21
    else:
        dataset_tag = " ".join(w.upper() for w in folder_name.split('_'))  # e.g. FS STATIC

    model_tag = os.path.splitext(os.path.basename(args.config_path))[0]
    size_tag  = f"{img_size[0]}x{img_size[1]}"

    results_file.write(
        f"{args.method} {model_tag} {args.combine_score} {dataset_tag}  {size_tag}  "
        f"AUPRC score:{prc_auc * 100.0}   FPR@TPR95:{fpr * 100.0}"
    )
    results_file.close()


if __name__ == '__main__':
    main()
