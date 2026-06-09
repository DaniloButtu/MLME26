import os
import sys
import zipfile
import torch
import numpy as np
import yaml
import importlib
import warnings
import math
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from iouEval import iouEval, getColorEntry

#########################################################################
# Set the main project directory path
# and add the 'eomt' folder which contains the model to the path.
#################################################################################
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'eomt'))

##################################################
# Define constants for the number of classes.
# MODEL_NUM_CLASSES = 19 because the model is trained on the 19 actual classes of Cityscapes (trainId 0-18).
# IOU_NUM_CLASSES   = 20 because the IoU metric also considers the void class at index 19.
######################################################################################################################
MODEL_NUM_CLASSES = 19
IOU_NUM_CLASSES   = 20

############################################################################
# Mapping from original Cityscapes labelId (0-255) to trainId (0-19).
# Create an array of 256 elements, all initialized to 19 (void).
# Then overwrite only the positions corresponding to the 19 classes of interest.
##############################################################################################
label2train = np.full(256, IOU_NUM_CLASSES - 1, dtype=np.uint8)   # default void (19)
train_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}
for lid, tid in train_mapping.items():
    label2train[lid] = tid

############################################################################################
# Custom dataset to read images and labels directly from ZIP files,
# without extracting them. Supports splits 'val', 'train', 'test', etc.
###############################################################################################
class ZipCityscapesDataset(Dataset):
    def __init__(self, img_zip_path, lbl_zip_path, split='val'):
        self.split = split
        # Open the two ZIP files in read mode
        self.img_zip = zipfile.ZipFile(img_zip_path, 'r')
        self.lbl_zip = zipfile.ZipFile(lbl_zip_path, 'r')

        # Build the image prefix for the requested split, e.g. 'leftImg8bit/val/'
        img_prefix = f'leftImg8bit/{split}/'
        self.samples = []
        # Iterate over all files inside the image ZIP
        for name in self.img_zip.namelist():
            # If the file is in the correct folder and ends with '_leftImg8bit.png'
            if name.startswith(img_prefix) and name.endswith('_leftImg8bit.png'):
                # Build the corresponding label name:
                # replace 'leftImg8bit' with 'gtFine' and change suffix from '_leftImg8bit.png' to '_gtFine_labelIds.png'
                lbl_name = name.replace('leftImg8bit/', 'gtFine/').replace(
                    '_leftImg8bit.png', '_gtFine_labelIds.png'
                )
                # If the label exists in the label ZIP, add the pair to the list
                if lbl_name in self.lbl_zip.NameToInfo:
                    self.samples.append((name, lbl_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, lbl_name = self.samples[idx]

        # Read the image bytes from the ZIP and convert to a PIL RGB image
        img_bytes = self.img_zip.read(img_name)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        # Convert to numpy array, permute dimensions from (H, W, C) to (C, H, W) and get a uint8 tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)   # [3, H, W], torch.uint8

        # Read the label bytes from the ZIP and convert to a PIL image (grayscale)
        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))
        # Convert to uint8 numpy array (contains original labelId 0-255)
        label = np.array(label, dtype=np.uint8)
        # Apply the labelId -> trainId mapping using the label2train array
        label = label2train[label]                                    # map to trainId
        # Convert to torch.long tensor and add a channel dimension (1, H, W)
        label = torch.from_numpy(label).long().unsqueeze(0)           # [1, H, W]

        return image, label

def main():
    # Command line argument parsing
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True, help='Model configuration .yaml file')
    parser.add_argument('--weights', required=True, help='EoMT model .bin weights')
    parser.add_argument('--img-zip', required=True, help='ZIP of leftImg8bit_trainvaltest.zip')
    parser.add_argument('--lbl-zip', required=True, help='ZIP of gtFine_trainvaltest.zip')
    parser.add_argument('--subset', default='val')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    #########################################
    # Load configuration from the YAML file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the crop window size from the config, default (640,640)
    img_size = config.get('data', {}).get('init_args', {}).get('img_size', (640, 640))

    #################################################################
    # Build the ENCODER (backbone) specified in the config
    #############################################################
    encoder_cfg = config['model']['init_args']['network']['init_args']['encoder']
    enc_mod, enc_cls = encoder_cfg['class_path'].rsplit('.', 1)   # split module and class name
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls)   # dynamic import
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get('init_args', {}))

    ################################################################
    # Build the network (decoder + already instantiated encoder)
    ########################################################
    net_cfg = config['model']['init_args']['network']
    net_mod, net_cls = net_cfg['class_path'].rsplit('.', 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls)
    # Filter arguments: exclude 'encoder' because we pass it separately
    network_kwargs = {k: v for k, v in net_cfg['init_args'].items() if k != 'encoder'}
    network = network_cls(masked_attn_enabled=False, num_classes=MODEL_NUM_CLASSES,
                          encoder=encoder, **network_kwargs)

    #########################################
    # Build the LIGHTNING MODULE
    ##################################
    lit_mod, lit_cls = config['model']['class_path'].rsplit('.', 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config['model']['init_args'].items() if k != 'network'}
    # If 'stuff_classes' exists in the data config, pass it to the module
    if 'stuff_classes' in config.get('data', {}).get('init_args', {}):
        model_kwargs['stuff_classes'] = config['data']['init_args']['stuff_classes']

    model = lit_cls(img_size=img_size, num_classes=MODEL_NUM_CLASSES,
                    network=network, **model_kwargs).eval().to(device)


    # Load weights from the .bin file
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    # Special handling for the backbone's pos_embed (transformer positional embedding).
    # If the dimensions in the checkpoint differ from the current model, interpolate to adapt.
    key = 'network.encoder.backbone.pos_embed'
    if key in state_dict:
        ckpt_pos = state_dict[key]
        model_pos = model.state_dict()[key]
        if ckpt_pos.shape != model_pos.shape:
            dim = ckpt_pos.shape[-1]   # embedding channel dimension
            # Calculate grid size in checkpoint and model
            ckpt_size = int(math.sqrt(ckpt_pos.shape[1]))
            model_size = int(math.sqrt(model_pos.shape[1]))
            # Reshape from (1, N, dim) to (1, H, W, dim) and permute to (1, dim, H, W)
            ckpt_pos_2d = ckpt_pos.reshape(1, ckpt_size, ckpt_size, dim).permute(0, 3, 1, 2)
            # Interpolate to the new size
            interp = F.interpolate(ckpt_pos_2d, size=(model_size, model_size),
                                   mode='bicubic', align_corners=False)
            # Return to original shape (1, N, dim)
            state_dict[key] = interp.permute(0, 2, 3, 1).reshape(1, model_size * model_size, dim)

    # Load the state_dict into the model, allowing missing keys (strict=False)
    model.load_state_dict(state_dict, strict=False)
    print("EoMT model and weights loaded successfully")

    #########################################################
    # Create DataLoader using the ZIP-based dataset
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Instance of the class that computes IoU (20 classes, ignore index 19)
    iou_eval = iouEval(IOU_NUM_CLASSES)

    # Evaluation loop
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            # Move images and labels to the device
            images = images.to(device)       # [B, 3, H, W] uint8
            labels = labels.to(device)

            # Convert the batch into a list of single tensors (for the windowed function)
            imgs_list = [img for img in images]      # list of tensors [3, H, W] uint8
            img_sizes = [img.shape[-2:] for img in imgs_list]   # original dimensions of each image

            #################################################################################################
            # Split into windows (crops) using the window_imgs_semantic method.
            # It automatically handles padding and returns crops and origin information.
            # If the model is wrapped in DataParallel, call the method on the underlying module.
            if isinstance(model, torch.nn.DataParallel):
                crops, origins = model.module.window_imgs_semantic(imgs_list)
            else:
                crops, origins = model.window_imgs_semantic(imgs_list)

            # Forward the crops through the model. Returns two tuples:
            # mask_logits_per_layer (list of tensors, one per layer) and class_logits_per_layer.
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            # Take only the output of the last layer (index 0)
            mask_logits = mask_logits_per_layer[0]
            class_logits = class_logits_per_layer[0]

            # Interpolate mask_logits to the img_size (the window size)
            mask_logits = F.interpolate(mask_logits, size=img_size,
                                        mode='bilinear', align_corners=False)

           
            # Convert mask_logits and class_logits to per-pixel logits (for each class)
            if isinstance(model, torch.nn.DataParallel):
                crop_logits = model.module.to_per_pixel_logits_semantic(mask_logits, class_logits)
                # Reassemble windows into the original image
                logits_list = model.module.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            else:
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            # logits_list is a list of tensors, each of shape (num_classes, H_orig, W_orig)
            # Take the first element of the batch (batch size = 1) and add a batch dimension
            logits = logits_list[0].unsqueeze(0)              # [1, 20, H, W]

            # Compute probabilities with softmax and then the predicted class (argmax)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1, keepdim=True)         # [1, 1, H, W]

            # Update the confusion matrix in the IoU evaluator
            iou_eval.addBatch(preds, labels)
            print(f"Processed batch {step}")

 
    # Compute and print IoU results
    iou_mean, iou_classes = iou_eval.getIoU()

    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
                   'pole', 'traffic light', 'traffic sign', 'vegetation',
                   'terrain', 'sky', 'person', 'rider', 'car',
                   'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    print("=======================================")
    for i, name in enumerate(class_names):
        color = getColorEntry(iou_classes[i].item())
        print(f"{color}{name:15s}: {iou_classes[i].item()*100:.2f}%\033[0m")
    print("=======================================")
    color = getColorEntry(iou_mean.item())
    print(f"{color}MEAN IoU: {iou_mean.item()*100:.2f}%\033[0m")

if __name__ == '__main__':
    main()