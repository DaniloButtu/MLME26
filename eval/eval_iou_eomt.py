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
# Imposto il percorso della directory principale del progetto
# e aggiungo al path la cartella 'eomt' che contiene il modello.
#################################################################################
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'eomt'))

##################################################
# Definisco le costanti per il numero di classi.
# MODEL_NUM_CLASSES = 19 perché il modello è addestrato sulle 19 classi effettive di Cityscapes (trainId 0-18).
# IOU_NUM_CLASSES = 20 perché la metrica IoU considera anche la classe void all'indice 19.
######################################################################################################################
MODEL_NUM_CLASSES = 19
IOU_NUM_CLASSES   = 20

############################################################################
# Mappatura da labelId originale di Cityscapes (0-255) a trainId (0-19).
# Creo un array di 256 elementi, inizializzati tutti a 19 (void).
# Poi sovrascrivo solo le posizioni corrispondenti alle 19 classi di interesse.
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
# Dataset personalizzato per leggere immagini ed etichette direttamente da file ZIP,
# senza estrarli. Supporta split 'val', 'train', 'test' ecc.
###############################################################################################
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
        # Converto in array numpy, permuto le dimensioni da (H, W, C) a (C, H, W) e ottengo un tensore uint8
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)   # [3, H, W], torch.uint8

        # Leggo i byte dell'etichetta dallo ZIP e li converto in immagine PIL (in scala di grigi)
        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))
        # Converto in array numpy uint8 (contiene i labelId originali 0-255)
        label = np.array(label, dtype=np.uint8)
        # Applico la mappatura labelId -> trainId usando l'array label2train
        label = label2train[label]                                    # mappa a trainId
        # Converto a tensore torch.long e aggiungo una dimensione per il canale (1, H, W)
        label = torch.from_numpy(label).long().unsqueeze(0)           # [1, H, W]

        return image, label

def main():
    # Parsing degli argomenti da riga di comando
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True, help='File .yaml di configurazione del modello')
    parser.add_argument('--weights', required=True, help='Pesi .bin del modello EoMT')
    parser.add_argument('--img-zip', required=True, help='ZIP delle immagini leftImg8bit_trainvaltest.zip')
    parser.add_argument('--lbl-zip', required=True, help='ZIP delle etichette gtFine_trainvaltest.zip')
    parser.add_argument('--subset', default='val')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    #########################################
    # Caricamento della configurazione dal file YAML
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Estraggo la dimensione delle finestre (crops) dalla configurazione, default (640,640)
    img_size = config.get('data', {}).get('init_args', {}).get('img_size', (640, 640))

    #################################################################
    # Costruzione dell'ENCODER (backbone) specificato nel config
    #############################################################
    encoder_cfg = config['model']['init_args']['network']['init_args']['encoder']
    enc_mod, enc_cls = encoder_cfg['class_path'].rsplit('.', 1)   # separo modulo e nome classe
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls)   # import dinamico
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get('init_args', {}))

    ################################################################
    # Costruzione del network (decoder + encoder già istanziato)
    ########################################################
    net_cfg = config['model']['init_args']['network']
    net_mod, net_cls = net_cfg['class_path'].rsplit('.', 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls)
    # Filtro gli argomenti: escludo 'encoder' perché lo passo separatamente
    network_kwargs = {k: v for k, v in net_cfg['init_args'].items() if k != 'encoder'}
    network = network_cls(masked_attn_enabled=False, num_classes=MODEL_NUM_CLASSES,
                          encoder=encoder, **network_kwargs)

    #########################################
    # Costruzione del LIGHTINGMODULE 
    ##################################
    lit_mod, lit_cls = config['model']['class_path'].rsplit('.', 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config['model']['init_args'].items() if k != 'network'}
    # Se esiste 'stuff_classes' nei dati, lo passo al modulo
    if 'stuff_classes' in config.get('data', {}).get('init_args', {}):
        model_kwargs['stuff_classes'] = config['data']['init_args']['stuff_classes']

    model = lit_cls(img_size=img_size, num_classes=MODEL_NUM_CLASSES,
                    network=network, **model_kwargs).eval().to(device)


    # Caricamento dei pesi dal file .bin
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    # Gestione speciale per il pos_embed del backbone (positional embedding dei transformer).
    # Se le dimensioni nel checkpoint differiscono da quelle del modello corrente, le interpolo per adattarle.
    key = 'network.encoder.backbone.pos_embed'
    if key in state_dict:
        ckpt_pos = state_dict[key]
        model_pos = model.state_dict()[key]
        if ckpt_pos.shape != model_pos.shape:
            dim = ckpt_pos.shape[-1]   # dimensione del canale di embedding
            # Calcolo la dimensione della griglia nel checkpoint e nel modello
            ckpt_size = int(math.sqrt(ckpt_pos.shape[1]))
            model_size = int(math.sqrt(model_pos.shape[1]))
            # Rimodello da (1, N, dim) a (1, H, W, dim) e permuto a (1, dim, H, W)
            ckpt_pos_2d = ckpt_pos.reshape(1, ckpt_size, ckpt_size, dim).permute(0, 3, 1, 2)
            # Interpolazione alla nuova dimensione
            interp = F.interpolate(ckpt_pos_2d, size=(model_size, model_size),
                                   mode='bicubic', align_corners=False)
            # Torno alla forma (1, N, dim) originale
            state_dict[key] = interp.permute(0, 2, 3, 1).reshape(1, model_size * model_size, dim)

    # Carico lo state_dict nel modello, consentendo chiavi mancanti (strict=False)
    model.load_state_dict(state_dict, strict=False)
    print("Modello e pesi EoMT caricati con successo")

    #########################################################
    # Creazione del DataLoader utilizzando il dataset da ZIP
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Istanza della classe che calcola l'IoU (20 classi, ignore index 19)
    iou_eval = iouEval(IOU_NUM_CLASSES)

    # Ciclo di valutazione 
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            # Sposto immagini e etichette sul dispositivo
            images = images.to(device)       # [B, 3, H, W] uint8
            labels = labels.to(device)

            # Converte il batch in lista di tensori singoli (per la funzione windowed)
            imgs_list = [img for img in images]      # lista di tensori [3, H, W] uint8
            img_sizes = [img.shape[-2:] for img in imgs_list]   # dimensioni originali di ogni immagine

            #################################################################################################
            # Suddivisione in finestre (crops) tramite il metodo window_imgs_semantic.
            # Gestisce automaticamente il padding e restituisce i crops e le informazioni di origine.
            # Se il modello è incapsulato in DataParallel, chiama il metodo sul modulo sottostante.
            if isinstance(model, torch.nn.DataParallel):
                crops, origins = model.module.window_imgs_semantic(imgs_list)
            else:
                crops, origins = model.window_imgs_semantic(imgs_list)

            # Forward dei crops attraverso il modello. Restituisce due tuple:
            # mask_logits_per_layer (lista di tensori, uno per layer) e class_logits_per_layer.
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            # Prendo solo l'output dell'ultimo layer (indice 0)
            mask_logits = mask_logits_per_layer[0]
            class_logits = class_logits_per_layer[0]

            # Interpolo i mask_logits per portarli alla dimensione img_size (quella delle finestre)
            mask_logits = F.interpolate(mask_logits, size=img_size,
                                        mode='bilinear', align_corners=False)

           
            # Conversione dei mask_logits e class_logits in logits per pixel (per ogni classe)
            if isinstance(model, torch.nn.DataParallel):
                crop_logits = model.module.to_per_pixel_logits_semantic(mask_logits, class_logits)
                # Ricompongo le finestre nell'immagine originale
                logits_list = model.module.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            else:
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            # logits_list è una lista di tensori, ognuno di shape (num_classes, H_orig, W_orig)
            # Prendo il primo elemento del batch (batch size = 1) e aggiungo dimensione batch
            logits = logits_list[0].unsqueeze(0)              # [1, 20, H, W]

            # Calcolo le probabilità con softmax e poi la classe predetta (argmax)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1, keepdim=True)         # [1, 1, H, W]

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