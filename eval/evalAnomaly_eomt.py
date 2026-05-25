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


# Path setup: add project root and the eomt folder to sys.path so that
# we can import the model modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
eomt_path = os.path.join(project_root, 'eomt')
if eomt_path not in sys.path:
    sys.path.append(eomt_path)


seed = 42
seed_everything(seed, verbose=False)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES  = 19   # Number of semantic classes (Cityscapes trainId 0-18)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _interpolate_pos_embed(state_dict, model, encoder_cfg, img_size):
    #To interpolate the positional embedding of the backbone when the checkpoint resolution differs from the current model.
    key = 'network.encoder.backbone.pos_embed'
    if key not in state_dict:
        return
    ckpt_pe  = state_dict[key]
    model_pe = model.state_dict().get(key)
    if model_pe is None or ckpt_pe.shape == model_pe.shape:
        return

    print(f"Interpolating pos_embed: {ckpt_pe.shape} -> {model_pe.shape}")
    dim        = ckpt_pe.shape[-1]                     # Embedding dimension
    patch_size = encoder_cfg.get("init_args", {}).get("patch_size", 16)
    target_h   = img_size[0] // patch_size            # Number of patches in height
    target_w   = img_size[1] // patch_size            # Number of patches in width
    ckpt_seq   = ckpt_pe.shape[1]                     # Original sequence length
    ckpt_h     = int(math.sqrt(ckpt_seq))             # square grid
    ckpt_w     = ckpt_seq // ckpt_h

    # Reshape to 2D grid, interpolate, then reshape back to 1D sequence
    pe_2d        = ckpt_pe.reshape(1, ckpt_h, ckpt_w, dim).permute(0, 3, 1, 2)
    interpolated = F.interpolate(pe_2d, size=(target_h, target_w),
                                 mode='bicubic', align_corners=False)
    state_dict[key] = interpolated.permute(0, 2, 3, 1).reshape(1, target_h * target_w, dim)


def _build_classic_map(mask_logits_per_layer, class_logits_per_layer,
                       revert_fn, origins, img_sizes, img_size, method):
    
    # To Reconstruct a per-pixel anomaly score from the semantic segmentation heads.
   
    # Take the output of the last decoder layer (only one layer when masked_attn is disabled)
    mask_logits  = mask_logits_per_layer[-1].float()   # [B, Q, H_tok, W_tok]
    class_logits = class_logits_per_layer[-1].float()  # [B, Q, C+1]  (C=19, +1 for void)

    # Upsample mask logits to the crop size (img_size) using bilinear interpolation
    mask_logits = F.interpolate(mask_logits, size=img_size,
                                mode="bilinear", align_corners=False)

    # Combine mask and class logits to obtain per‑pixel class scores.
    # Formula: sigmoid(mask) * softmax(class)[..., :-1]  (exclude void)
    crop_logits = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )

    # Stitch the overlapping crops back into full‑resolution images.
    # revert_fn is either revert_window_logits (for anomaly module) or
    # revert_window_logits_semantic (for standard module).
    logits_list = revert_fn(crop_logits, origins, img_sizes)
    logits      = logits_list[0].unsqueeze(0)          # [1, C, H, W]

    # Compute class probabilities via softmax over the semantic classes (no void)
    probs = torch.softmax(logits, dim=1)               # [1, C, H, W]

    
    # Anomaly score according to the selected method
    if method == "msp":
        # Maximum Softmax Probability: anomaly = 1 - max(prob)
        cmap = 1.0 - torch.max(probs, dim=1)[0].squeeze().cpu().numpy()
    elif method == "max_logit":
        # Max Logit: anomaly = - max(logit)  (higher logit -> more confident -> less anomalous)
        cmap = -np.max(logits.squeeze(0).cpu().numpy(), axis=0)
    elif method == "max_entropy":
        # Entropy of the probability distribution: higher entropy = more uncertain = anomaly
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        cmap = entropy.squeeze(0).cpu().numpy()
    elif method == "rba":
        # rba – sum of tanh(logit) over classes, then negate
        tanh_sum = np.sum(np.tanh(logits.squeeze(0).cpu().numpy()), axis=0)
        cmap = -tanh_sum
    else:
        raise ValueError(f"Unknown method: {method}")

    return cmap


# Main evaluation routine
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

    # Build an absolute default path for the config using project_root
    abs_default_config = os.path.join(project_root, "eomt/configs/dinov2/cityscapes/semantic/eomt_mlp.yaml")

    parser.add_argument(
        "--config_path",
        default=abs_default_config,
        help="Path to the YAML config that defines model + data.",
    )

    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Path to the checkpoint to evaluate. "
             "If omitted, the script uses ckpt_path from config or no pretrained weights.",
    )
    parser.add_argument('--subset',      default="val")
    parser.add_argument('--num-workers',  type=int, default=4)   # Not used in this script (no DataLoader)
    parser.add_argument('--batch-size',   type=int, default=1)   # Not used
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

    anomaly_score_list = []   # Will store the anomaly map for each image
    ood_gts_list       = []   # Will store the corresponding ground truth masks

    # Open (or create) a results file to append metrics
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    results_file = open('results.txt', 'a')

    #####################################
    # Load the YAML configuration file
    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')


    # Detect operating mode from the Lightning module class_path
    lit_class_path = config["model"]["class_path"]
    # True  -> AnomalyClassificationModule  (eomt_mlp.yaml)
    # False -> MaskClassificationSemantic   (eomt_base_640.yaml)
    is_anomaly_module = "AnomalyClassificationModule" in lit_class_path
    print(f"Mode: {'AnomalyClassificationModule' if is_anomaly_module else 'MaskClassificationSemantic'}")

    #####################################################################
    # Build the ENCODER (backbone) as specified in the config
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

    # Determine the image size (crop size) for the encoder.
    # Priority: command line override > encoder config > model config > default (1024,1024)
    enc_kwargs = encoder_cfg.get("init_args", {}).copy()
    img_size   = enc_kwargs.get(
        "img_size",
        config.get("model", {}).get("init_args", {}).get("img_size", [1024, 1024])
    )
    if isinstance(img_size, list):
        img_size = tuple(img_size)

    if args.img_size is not None:
        img_size = tuple(args.img_size)
        print(f"img_size overridden by --img_size: {img_size}")
    else:
        print(f"img_size from config: {img_size}")

    enc_kwargs["img_size"] = list(img_size)   # Update encoder kwargs
    encoder = encoder_cls(**enc_kwargs)


    #################################################
    # Build the EoMT NETWORK (decoder + encoder)
    ############
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)

    # Filter out 'encoder' from init_args because we pass it separately
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network_kwargs["masked_attn_enabled"] = False   # Disable masked attention for inference
    network_kwargs["num_classes"]         = NUM_CLASSES

    network = network_cls(encoder=encoder, **network_kwargs)

    #######################################################################
    # Build the LIGHTINGMODULE (either AnomalyClassificationModule or
    # MaskClassificationSemantic)
    #################################
    lit_module_name, lit_class_name = lit_class_path.rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)

    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]
    model_kwargs["img_size"] = img_size

    # The anomaly module does not accept a 'num_classes' argument, the semantic one does.
    if is_anomaly_module:
        model_kwargs.pop("num_classes", None)
    else:
        model_kwargs.setdefault("num_classes", NUM_CLASSES)

    model_kwargs["ckpt_path"] = None   # We will load weights manually
    model = lit_cls(network=network, **model_kwargs).eval().to(device)

    ########################################
    # Load the checkpoint (weights)
    ckpt_path = args.ckpt_path or config.get("ckpt_path") or None

    if ckpt_path:
        print(f"Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove compilation artifacts if present
        state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}

        # Interpolate positional embedding if needed
        _interpolate_pos_embed(state_dict, model, encoder_cfg, img_size)

        # Load with strict=False because the anomaly head may be missing in some configs
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


    # Inference loop over all images matching the input pattern
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(f"\nProcessing: {path}")

        # Load and preprocess the image
        img_pil    = Image.open(path).convert('RGB')
        img_np     = np.array(img_pil)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

        # Forward pass through the model
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if not args.cpu else "cpu"):
            # The anomaly module expects float images (will divide by 255 later),
            # while the semantic module expects uint8 (division inside window_imgs_semantic).
            if is_anomaly_module:
                imgs_list = [img_tensor.float()]
            else:
                imgs_list = [img_tensor]

            img_sizes = [img.shape[-2:] for img in imgs_list]   # Original image dimensions

            # Obtain the model's underlying module
            mod = model.module if isinstance(model, torch.nn.DataParallel) else model

            # Split image(s) into overlapping windows (crops) ready for the model.
            if is_anomaly_module:
                crops, origins = mod.window_imgs(imgs_list)
            else:
                crops, origins = mod.window_imgs_semantic(imgs_list)

            # Normalise pixel values to [0,1] (the model expects this)
            x = crops.float() / 255.0

            # Forward through the network (encoder + decoder)
            # Returns: mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer
            mask_logits_per_layer, class_logits_per_layer, anomaly_logits_per_layer = model.network(x)

            # Check if an anomaly head is present and produced valid output
            has_anomaly_head = (
                is_anomaly_module
                and anomaly_logits_per_layer[-1] is not None
            )

            #######################################################
            # Process MLP anomaly head output (if available)
            if has_anomaly_head:
                crop_anomaly = anomaly_logits_per_layer[-1]
                crop_anomaly = F.interpolate(crop_anomaly, size=img_size,
                                             mode="bilinear", align_corners=False)
                # Stitch crops and apply sigmoid to obtain anomaly probability map
                anomaly_list     = mod.revert_window_logits(crop_anomaly, origins, img_sizes)
                mlp_anomaly_map  = torch.sigmoid(anomaly_list[0]).squeeze(0).cpu().numpy()
                mlp_anomaly_map  = np.nan_to_num(mlp_anomaly_map,
                                                  nan=0.0, posinf=1.0, neginf=0.0)

            
            # Build classic anomaly map from semantic segmentation
            if is_anomaly_module:
                revert_fn = mod.revert_window_logits          # For single‑channel anomaly maps
            else:
                revert_fn = mod.revert_window_logits_semantic # For multi‑class logits

            classic_map = _build_classic_map(
                mask_logits_per_layer, class_logits_per_layer,
                revert_fn, origins, img_sizes, img_size, args.method
            )

            
            # Combine the two anomaly maps if both exist
            if has_anomaly_head:
                # Normalise the classic map to [0,1] per image (min‑max scaling)
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
                anomaly_map = classic_map

            anomaly_result = anomaly_map

            ##########################################################################
            # Debug: save a heatmap for the first image to visualise anomalies
            if len(ood_gts_list) <= 0:
                debug_stem = os.path.splitext(os.path.basename(path))[0]
                map_u8     = cv2.normalize(anomaly_map, None, 0, 255,
                                           cv2.NORM_MINMAX, cv2.CV_8U)
                heatmap    = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)
                debug_name = f"debug_heatmap_{debug_stem}.jpg"
                cv2.imwrite(debug_name, heatmap)
                print(f"\n--- Debug heatmap saved as {debug_name} ---\n")

        # Load and process the ground truth mask for the current image
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

        
        # Remap ground truth labels to a binary format (0 = in-distribution,
        # 1 = anomaly) for each dataset.
        if "RoadAnomaly" in pathGT:
            # In RoadAnomaly, label 2 indicates anomaly -> map to 1
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)
        if "LostAndFound" in pathGT:
            # LostAndFound: 0 = void/ignore, 1 = road, 2..200 = anomalies
            ood_gts = np.where(ood_gts == 0,  255, ood_gts)   # 255 = ignore
            ood_gts = np.where(ood_gts == 1,  0,   ood_gts)   # road -> 0
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)  # anomalies -> 1
        if "Streethazard" in pathGT:
            # Streethazard: 14 = anomaly, anything <20 = road/other -> 0
            ood_gts = np.where(ood_gts == 14, 255, ood_gts)
            ood_gts = np.where(ood_gts < 20,  0,   ood_gts)
            ood_gts = np.where(ood_gts == 255, 1,   ood_gts)

        # Skip images that contain no anomaly pixels (no ground truth positive)
        if 1 not in np.unique(ood_gts):
            continue

        # Store the anomaly map and the corresponding ground truth
        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        # Clean up to free memory
        del anomaly_result, ood_gts, mask, img_tensor, img_pil, img_np
        torch.cuda.empty_cache()

    results_file.write("\n")

    if not ood_gts_list:
        print("No valid ground truths found.")
        results_file.close()
        return

    ##########################################
    # Compute anomaly detection metrics
    ood_gts        = np.array(ood_gts_list)          # List of 2D masks
    anomaly_scores = np.array(anomaly_score_list)    # List of 2D anomaly maps

    # Flatten all pixels while keeping only those marked as inlier (0) or anomaly (1)
    valid_mask = (ood_gts == 0) | (ood_gts == 1)
    val_out    = anomaly_scores[valid_mask]
    val_label  = ood_gts[valid_mask]

    # AUPRC
    prc_auc = average_precision_score(val_label, val_out)
    # FPR at 95% TPR
    fpr     = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0:.4f}')
    print(f'FPR@TPR95:   {fpr   * 100.0:.4f}')


    # Create a tag for the dataset and model for the results file
    input_parts  = re.split(r'[\\/]', str(args.input[0]))
    try:
        ds_idx      = [p.lower() for p in input_parts].index('dataset')
        folder_name = input_parts[ds_idx + 1] if ds_idx + 1 < len(input_parts) else input_parts[0]
    except ValueError:
        folder_name = input_parts[0]

    caps_digits = "".join(re.findall(r'[A-Z0-9]', folder_name))
    if caps_digits:
        dataset_tag = caps_digits
    else:
        dataset_tag = " ".join(w.upper() for w in folder_name.split('_'))

    model_tag = os.path.splitext(os.path.basename(args.config_path))[0]
    size_tag  = f"{img_size[0]}x{img_size[1]}"

    # Append the results to the file
    results_file.write(
        f"{args.method} {model_tag} {args.combine_score} {dataset_tag}  {size_tag}  "
        f"AUPRC score:{prc_auc * 100.0}   FPR@TPR95:{fpr * 100.0}"
    )
    results_file.close()


if __name__ == '__main__':
    main()