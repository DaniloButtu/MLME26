import os
import zipfile
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor
from erfnet import ERFNet          # assicurati che erfnet.py sia nel path
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20   # 19 classi valide (trainId 0-18) + 1 void (indice 19)

# Mappatura labelId -> trainId (Cityscapes)
label2train = np.full(256, 19, dtype=np.uint8)
train_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}
for lid, tid in train_mapping.items():
    label2train[lid] = tid

class ZipCityscapesDataset(Dataset):
    def __init__(self, img_zip_path, lbl_zip_path, split='val'):
        self.split = split
        
        self.img_zip = zipfile.ZipFile(img_zip_path, 'r')
        self.lbl_zip = zipfile.ZipFile(lbl_zip_path, 'r')

        # Elenco tutti i file immagine per lo split (batch) richiesto
        img_prefix = f'leftImg8bit/{split}/'
        self.samples = []
        for name in self.img_zip.namelist():
            if name.startswith(img_prefix) and name.endswith('_leftImg8bit.png'):
                # Il corrispondente file etichetta: stesso percorso ma in gtFine/split/... e suffisso _gtFine_labelIds.png
                lbl_name = name.replace('leftImg8bit/', 'gtFine/').replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                # Verifica che esista nel secondo ZIP
                if lbl_name in self.lbl_zip.NameToInfo:
                    self.samples.append((name, lbl_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, lbl_name = self.samples[idx]

        # Leggo l'immagine dal primo ZIP
        img_bytes = self.img_zip.read(img_name)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')

        # Leggo l'etichetta dal secondo ZIP
        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))   # modalità L, labelId originali

        # Ridimensionamento (ERFNet standard 512x1024)
        image = resize(image, (512, 1024), interpolation=Image.BILINEAR)
        label = resize(label, (512, 1024), interpolation=Image.NEAREST)

        image = to_tensor(image)                     # [3, H, W] float32

        label = np.array(label, dtype=np.uint8)
        label = label2train[label]                   # mappa a trainId (19 = void)
        label = torch.from_numpy(label).long().unsqueeze(0)  # [1, H, W]

        return image, label

def main():
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default='/content/drive/MyDrive/ML/trained_models/')
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth')
    parser.add_argument('--img-zip', required=True,
                        help='Percorso al file ZIP delle immagini (leftImg8bit_trainvaltest.zip)')
    parser.add_argument('--lbl-zip', required=True,
                        help='Percorso al file ZIP delle etichette (gtFine_trainvaltest.zip)')
    parser.add_argument('--subset', default='val')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Caricamento ERFNet
    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    state_dict = torch.load(weightspath, map_location='cpu')

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(f"{name} non caricato")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, state_dict)
    model.eval()
    print("Modello e pesi ERFNet caricati con successo")

    # Dataset da ZIP
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    iou_eval = iouEval(NUM_CLASSES)

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            if not args.cpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            preds = outputs.max(dim=1)[1].unsqueeze(1)   # [B, 1, H, W]

            iou_eval.addBatch(preds, labels)
            print(f"Processato batch {step}")

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