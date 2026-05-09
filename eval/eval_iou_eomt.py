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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'eomt'))

MODEL_NUM_CLASSES = 19      # classi reali di cityescapes
IOU_NUM_CLASSES   = 20      # classi totali in uscita (incluso void all'indice 19)

# Mappatura Cityscapes labelId -> trainId
label2train = np.full(256, IOU_NUM_CLASSES - 1, dtype=np.uint8)   # default void (19)
train_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}
for lid, tid in train_mapping.items():
    label2train[lid] = tid

class ZipCityscapesDataset(Dataset):
    # Per leggere immagini e label direttamente dai dataset zippati di cityescapes
    def __init__(self, img_zip_path, lbl_zip_path, split='val'):
        self.split = split
        self.img_zip = zipfile.ZipFile(img_zip_path, 'r')
        self.lbl_zip = zipfile.ZipFile(lbl_zip_path, 'r')

        img_prefix = f'leftImg8bit/{split}/'
        self.samples = []
        for name in self.img_zip.namelist():
            if name.startswith(img_prefix) and name.endswith('_leftImg8bit.png'):
                lbl_name = name.replace('leftImg8bit/', 'gtFine/').replace(
                    '_leftImg8bit.png', '_gtFine_labelIds.png'
                )
                if lbl_name in self.lbl_zip.NameToInfo:
                    self.samples.append((name, lbl_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, lbl_name = self.samples[idx]

        img_bytes = self.img_zip.read(img_name)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)   # [3, H, W], torch.uint8

        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))
        label = np.array(label, dtype=np.uint8)
        label = label2train[label]                                    # mappa a trainId
        label = torch.from_numpy(label).long().unsqueeze(0)           # [1, H, W]

        return image, label

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True, help='File .yaml di configurazione')
    parser.add_argument('--weights', required=True, help='Pesi .bin (es. eomt_cityscapes.bin)')
    parser.add_argument('--img-zip', required=True, help='ZIP immagini leftImg8bit_trainvaltest.zip')
    parser.add_argument('--lbl-zip', required=True, help='ZIP etichette gtFine_trainvaltest.zip')
    parser.add_argument('--subset', default='val')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Caricamento dei file config e del modello EoMT
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    img_size = config.get('data', {}).get('init_args', {}).get('img_size', (640, 640))

    # Encoder
    encoder_cfg = config['model']['init_args']['network']['init_args']['encoder']
    enc_mod, enc_cls = encoder_cfg['class_path'].rsplit('.', 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get('init_args', {}))

    # Network (con MODEL_NUM_CLASSES = 19)
    net_cfg = config['model']['init_args']['network']
    net_mod, net_cls = net_cfg['class_path'].rsplit('.', 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls)
    network_kwargs = {k: v for k, v in net_cfg['init_args'].items() if k != 'encoder'}
    network = network_cls(masked_attn_enabled=False, num_classes=MODEL_NUM_CLASSES,
                          encoder=encoder, **network_kwargs)

    # Lightning module
    lit_mod, lit_cls = config['model']['class_path'].rsplit('.', 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config['model']['init_args'].items() if k != 'network'}
    if 'stuff_classes' in config.get('data', {}).get('init_args', {}):
        model_kwargs['stuff_classes'] = config['data']['init_args']['stuff_classes']

    model = lit_cls(img_size=img_size, num_classes=MODEL_NUM_CLASSES,
                    network=network, **model_kwargs).eval().to(device)

    # Caricamento pesi
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    # Interpolazione del pos_embed se necessario
    key = 'network.encoder.backbone.pos_embed'
    if key in state_dict:
        ckpt_pos = state_dict[key]
        model_pos = model.state_dict()[key]
        if ckpt_pos.shape != model_pos.shape:
            dim = ckpt_pos.shape[-1]
            ckpt_size = int(math.sqrt(ckpt_pos.shape[1]))
            model_size = int(math.sqrt(model_pos.shape[1]))
            ckpt_pos_2d = ckpt_pos.reshape(1, ckpt_size, ckpt_size, dim).permute(0, 3, 1, 2)
            interp = F.interpolate(ckpt_pos_2d, size=(model_size, model_size),
                                   mode='bicubic', align_corners=False)
            state_dict[key] = interp.permute(0, 2, 3, 1).reshape(1, model_size * model_size, dim)

    model.load_state_dict(state_dict, strict=False)
    print("Modello e pesi EoMT caricati con successo")

    # DataLoader da ZIP
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    iou_eval = iouEval(IOU_NUM_CLASSES)   # 20 classi, ignoreIndex=19

    # Ciclo di valutazione
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)       # [B, 3, H, W] uint8
            labels = labels.to(device)

            imgs_list = [img for img in images]      # lista di tensori [3, H, W] uint8
            img_sizes = [img.shape[-2:] for img in imgs_list]

            # Inferenza windowed (gestisce automaticamente la divisione in crops)
            if isinstance(model, torch.nn.DataParallel):
                crops, origins = model.module.window_imgs_semantic(imgs_list)
            else:
                crops, origins = model.window_imgs_semantic(imgs_list)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = mask_logits_per_layer[0]
            class_logits = class_logits_per_layer[0]

            mask_logits = F.interpolate(mask_logits, size=img_size,
                                        mode='bilinear', align_corners=False)

            if isinstance(model, torch.nn.DataParallel):
                crop_logits = model.module.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.module.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            else:
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            logits = logits_list[0].unsqueeze(0)              # [1, 20, H, W]
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1, keepdim=True)         # [1, 1, H, W]

            iou_eval.addBatch(preds, labels)
            print(f"Processato batch {step}")

    iou_mean, iou_classes = iou_eval.getIoU()

    # Stampa risultati iou
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