import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from PIL import Image
import numpy as np

class OODAnomalyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(1024, 2048)):
        """
        Args:
            img_dir (str): Percorso della cartella contenente le immagini
            mask_dir (str): Percorso della cartella contenente le maschere
            img_size (tuple): Dimensione a cui fare il resize base delle immagini
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # Cerca tutti i file immagine (puoi rimuovere la condizione startswith se hanno nomi diversi)
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.startswith("anomaly_")])
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # Ricava il nome della maschera associata (es. anomaly_0000.png -> mask_0000.png)
        # Se hanno ESATTAMENTE lo stesso nome (es. img_00.png sia per l'immagine che per la maschera), 
        # puoi semplicemente usare: mask_name = img_name
        mask_name = img_name.replace("anomaly_", "mask_") 
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Carica immagine (RGB) e maschera (Grayscale)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Resize opzionale
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Converti immagine in tensore [C, H, W]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        
        # Binarizza la maschera (tutto ciò che è > 127 diventa 1, altrimenti 0)
        mask_np = np.array(mask)
        binary_mask = (mask_np > 127).astype(np.uint8)
        
        # Formatta il target per la tua anomaly_head
        target = {
            "masks": torch.from_numpy(binary_mask).unsqueeze(0).bool(), # Shape [1, H, W]
            "labels": torch.tensor([19], dtype=torch.long),             # 19 è la classe anomalia
            "is_crowd": torch.tensor([False], dtype=torch.bool)
        }
        
        return img_tensor, target

class AnomalyDataModule(pl.LightningDataModule):
    def __init__(self, 
        path="./anomaly_training/",  # <-- Rinominato da 'data_dir' a 'path' per coincidere col YAML
        batch_size=2, 
        num_workers=4, 
        img_size=(1024, 2048),
        num_classes=19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True
    ):
        super().__init__()
        self.data_dir = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
    def setup(self, stage=None):
        # Definisci i percorsi per le cartelle separate di Train
        train_img_dir = os.path.join(self.data_dir, "train", "images")
        train_mask_dir = os.path.join(self.data_dir, "train", "masks")
        
        # Definisci i percorsi per le cartelle separate di Validation
        val_img_dir = os.path.join(self.data_dir, "val", "images")
        val_mask_dir = os.path.join(self.data_dir, "val", "masks")
        
        # Inizializza i dataset passando le due cartelle distinte
        self.train_dataset = OODAnomalyDataset(train_img_dir, train_mask_dir, img_size=self.img_size)
        self.val_dataset = OODAnomalyDataset(val_img_dir, val_mask_dir, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, list(targets)