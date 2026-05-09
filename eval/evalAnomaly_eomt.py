# Copyright (c) OpenMMLab. All rights reserved.
import os
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

from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
eomt_path = os.path.join(project_root, 'eomt')
if eomt_path not in sys.path:
    sys.path.append(eomt_path)
seed = 42

# general reproducibility
seed_everything(seed, verbose=False)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 19
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--config_path', default="/content/MLME26/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml", help="Path to yaml config file")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--method', default='rba', choices=['msp', 'max_logit', 'max_entropy', 'rba'])
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    
    # LOAD ENCODER AND NETWORK
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # 1. Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    img_size = config.get("data", {}).get("init_args", {}).get("img_size", (640, 640))
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # 2. Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )

    # 3. Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]


    # LOAD LOCAL WEIGHTS ONLY
    local_bin = "drive/MyDrive/ML/eomt/bin/eomt_cityscapes.bin"
    print(f"Loading local weights from: {local_bin}")

    # Instantiate the model
    model = lit_cls(
        img_size=img_size,
        num_classes=NUM_CLASSES,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    # Load local state dict
    state_dict = torch.load(local_bin, map_location=device, weights_only=True)

    # Fix positional embedding shape mismatch if loading 1024 weights into 640 config
    key = 'network.encoder.backbone.pos_embed'
    if key in state_dict:
        ckpt_pos_embed = state_dict[key]
        model_pos_embed = model.state_dict()[key]
        
        if ckpt_pos_embed.shape != model_pos_embed.shape:
            print(f"Interpolating pos_embed from {ckpt_pos_embed.shape} to {model_pos_embed.shape}")
            dim = ckpt_pos_embed.shape[-1]
            
            ckpt_size = int(math.sqrt(ckpt_pos_embed.shape[1])) 
            model_size = int(math.sqrt(model_pos_embed.shape[1]))
            
            ckpt_pos_embed_2d = ckpt_pos_embed.reshape(1, ckpt_size, ckpt_size, dim).permute(0, 3, 1, 2)
            interpolated = F.interpolate(ckpt_pos_embed_2d, size=(model_size, model_size), mode='bicubic', align_corners=False)
            state_dict[key] = interpolated.permute(0, 2, 3, 1).reshape(1, model_size * model_size, dim)

    model.load_state_dict(state_dict, strict=False)
    print("Local model weights LOADED successfully.")

    
    # INFERENCE & EVALUATION LOOP
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        # 1. Apro l'immagine con PIL
        img_pil = Image.open(path).convert('RGB')
        
        # 2. Converto in numpy array
        img_np = np.array(img_pil)
        
        # 3. Converto in tensore PyTorch e sposto i canali per formare [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
        
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if not args.cpu else "cpu"):
            
            imgs_list = [img_tensor] 
            img_sizes = [img.shape[-2:] for img in imgs_list]
            
            if isinstance(model, torch.nn.DataParallel):
                crops, origins = model.module.window_imgs_semantic(imgs_list)
            else:
                crops, origins = model.window_imgs_semantic(imgs_list)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            
            mask_logits = mask_logits_per_layer[0]
            class_logits = class_logits_per_layer[0]
            
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, 
                size=img_size,
                mode="bilinear", 
                align_corners=False
            )
            
            if isinstance(model, torch.nn.DataParallel):
                crop_logits = model.module.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.module.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            else:
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            
            logits = logits_list[0].unsqueeze(0)
            sums = logits.sum(dim=1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-2) or logits.min() < 0.0:
              probs = torch.softmax(logits, dim=1)
            else:
              probs = logits
            void_probs = probs[:, -1, :, :].squeeze(0).cpu().numpy()
            logits_no_void = logits[:, :-1, :, :]


            if args.method == "msp":
              anomaly_map = 1.0 - torch.max(probs, dim=1)[0].squeeze().cpu().numpy()
            elif args.method == "max_logit":
              anomaly_map = -np.max(logits_no_void.squeeze(0).cpu().numpy(), axis=0)
        
            elif args.method == "max_entropy":
              entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
              anomaly_map = entropy.squeeze(0).cpu().numpy()
        
            elif args.method == "rba":
              anomaly_map = void_probs

        
            else:
              raise ValueError(f"Unknown method: {args.method}")
            anomaly_result = anomaly_map
        
            if len(ood_gts_list) == 0: 
              print(path)
              map_normalized = cv2.normalize(anomaly_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
              heatmap_img = cv2.applyColorMap(map_normalized, cv2.COLORMAP_JET)
              debug_name = "debug_heatmap.jpg"
              cv2.imwrite(debug_name, heatmap_img)
              print(f"\n--- MAPPA DI DEBUG SALVATA COME {debug_name} ---")
              print("Guardala per capire cosa la rete sta marcando come anomalia!\n")

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        try:
            mask = Image.open(pathGT)
        except Exception as e:
            print(f"Skipping {path}, GT not found.")
            continue

        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
             
        del anomaly_result, ood_gts, mask, img_tensor, img_pil, img_np
        torch.cuda.empty_cache()

    file.write( "\n")

    if not ood_gts_list:
        print("No valid Ground Truths found.")
        file.close()
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write((str(args.method) + '    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()