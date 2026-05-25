import os
import zipfile
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor
from erfnet import ERFNet          # importo il modello ERFNet 
from iouEval import iouEval, getColorEntry


# Definisco il numero di classi per l'IoU.
# NUM_CLASSES = 20 perché la metrica considera 19 classi reali (trainId 0-18) più il void all'indice 19.
# Il modello ERFNet è addestrato su 20 canali in uscita, l'ultimo viene ignorato nel calcolo dell'IoU.
NUM_CLASSES = 20

##############################################################################
# Mappatura da labelId originale di Cityscapes (0-255) a trainId (0-19).
# Creo un array di 256 elementi, inizializzati tutti a 19 (void).
# Poi sovrascrivo solo le posizioni corrispondenti alle 19 classi di interesse.
#########################################################################################
label2train = np.full(256, 19, dtype=np.uint8)
train_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}
for lid, tid in train_mapping.items():
    label2train[lid] = tid

#####################################################################################
# Dataset personalizzato per leggere immagini ed etichette direttamente da file ZIP,
# senza estrarli. Supporta split 'val', 'train', 'test' ecc.
# A differenza della versione per EoMT, qui le immagini e le etichette vengono ridimensionate forzatamente a 512×1024 (formato standard per ERFNet su Cityscapes).
class ZipCityscapesDataset(Dataset):
    def __init__(self, img_zip_path, lbl_zip_path, split='val'):
        self.split = split
        # Apro i due file ZIP in modalità lettura
        self.img_zip = zipfile.ZipFile(img_zip_path, 'r')
        self.lbl_zip = zipfile.ZipFile(lbl_zip_path, 'r')

        # Costruisco il prefisso delle immagini per lo split richiesto, es. 'leftImg8bit/val/'
        img_prefix = f'leftImg8bit/{split}/'
        self.samples = []
        # Scorro tutti i file all'interno dello ZIP delle immagini
        for name in self.img_zip.namelist():
            # Se il file è nella cartella giusta e termina con '_leftImg8bit.png'
            if name.startswith(img_prefix) and name.endswith('_leftImg8bit.png'):
                # Costruisco il nome corrispondente dell'etichetta:
                # sostituisco 'leftImg8bit' con 'gtFine' e cambio suffisso da '_leftImg8bit.png' a '_gtFine_labelIds.png'
                lbl_name = name.replace('leftImg8bit/', 'gtFine/').replace(
                    '_leftImg8bit.png', '_gtFine_labelIds.png'
                )
                # Se l'etichetta esiste nello ZIP delle etichette, aggiungo la coppia alla lista
                if lbl_name in self.lbl_zip.NameToInfo:
                    self.samples.append((name, lbl_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, lbl_name = self.samples[idx]

        # Leggo i byte dell'immagine dallo ZIP e li converto in un'immagine PIL RGB
        img_bytes = self.img_zip.read(img_name)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')

        # Leggo i byte dell'etichetta dallo ZIP e li converto in immagine PIL (in scala di grigi)
        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))

        # Ridimensionamento a 512x1024 (formato standard per ERFNet su Cityscapes)
        image = resize(image, (512, 1024), interpolation=Image.BILINEAR)
        label = resize(label, (512, 1024), interpolation=Image.NEAREST)

        # RECALL: to_tensor converte l'immagine in tensore float32 e divide automaticamente i pixel per 255 (porta in [0,1])
        image = to_tensor(image)                     # [3, 512, 1024] float32

        # Converto l'etichetta in array numpy uint8, applico la mappatura labelId -> trainId,
        # converto a tensore torch.long e aggiungo una dimensione per il canale (1, H, W)
        label = np.array(label, dtype=np.uint8)
        label = label2train[label]
        label = torch.from_numpy(label).long().unsqueeze(0)  # [1, 512, 1024]

        return image, label

def main():
    # Parsing degli argomenti da riga di comando
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default='/content/MLME26/trained_models/',
                        help='Directory dove si trovano i pesi')
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth',
                        help='Nome del file dei pesi (es. erfnet_pretrained.pth)')
    parser.add_argument('--img-zip', required=True,
                        help='Percorso al file ZIP delle immagini (leftImg8bit_trainvaltest.zip)')
    parser.add_argument('--lbl-zip', required=True,
                        help='Percorso al file ZIP delle etichette (gtFine_trainvaltest.zip)')
    parser.add_argument('--subset', default='val',
                        help='Split da valutare: val, train, test')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Numero di processi per il DataLoader')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Dimensione del batch (ERFNet processa intere immagini, quindi può essere >1)')
    parser.add_argument('--cpu', action='store_true',
                        help='Forza l\'esecuzione su CPU anche se CUDA è disponibile')
    args = parser.parse_args()

    # Determino il dispositivo (CUDA se disponibile e non richiesto esplicitamente la CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    #########################################################
    # Istanzio il modello ERFNet con 20 classi
    ##############################################
    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    # Costruisco il percorso completo del file dei pesi
    weightspath = os.path.join(args.loadDir, args.loadWeights)
    # Carico lo state_dict 
    state_dict = torch.load(weightspath, map_location='cpu')

    # Funzione per caricare lo state_dict rimuovendo eventuale prefisso 'module.' presente nei checkpoint
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # Se il nome inizia con 'module.', provo a rimuovere il prefisso
                if name.startswith("module."):
                    stripped_name = name.split("module.")[-1]
                    if stripped_name in own_state:
                        own_state[stripped_name].copy_(param)
                    else:
                        print(f"{stripped_name} non trovato nel modello")
                else:
                    print(f"{name} non caricato")
                continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, state_dict)
    model.eval()   # Imposto la modalità di valutazione 
    print("Modello e pesi ERFNet caricati con successo")

    ###########################################################
    # Creazione del DataLoader utilizzando il dataset da ZIP
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Istanza della classe che calcola l'IoU (20 classi, ignore index 19)
    iou_eval = iouEval(NUM_CLASSES)

    # Ciclo di valutazione 
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            # Sposto immagini e etichette sul dispositivo
            if not args.cpu:
                images = images.cuda()
                labels = labels.cuda()

            # Forward dell'intera immagine (dimensioni 512x1024)
            outputs = model(images)                     # shape [B, 20, 512, 1024]
            # Estraggo la classe con probabilità massima lungo la dimensione dei canali (dim=1)
            # e aggiungo una dimensione per il canale (keepdim=False, poi unsqueeze)
            preds = outputs.max(dim=1)[1].unsqueeze(1)   # [B, 1, 512, 1024]

            # Aggiorno la matrice di confusione nel valutatore IoU
            iou_eval.addBatch(preds, labels)
            print(f"Processato batch {step}")

    # Calcolo e stampa dei risultati IoU
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