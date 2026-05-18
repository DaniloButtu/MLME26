import os
import sys
import warnings
# Ignora specificamente il warning relativo a wandb.require
warnings.filterwarnings("ignore", message=".*wandb.require.*")
# Aggiungiamo i percorsi al sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
eomt_path = os.path.join(project_root, 'eomt')
if eomt_path not in sys.path:
    sys.path.append(eomt_path)

from lightning.pytorch.cli import LightningCLI
import torch
torch.set_float32_matmul_precision('medium')

# Importiamo i moduli in modo che LightningCLI possa trovarli tramite class_path
import eomt.training.mask_classification_semantic
import eomt.training.mask_classification_panoptic
import eomt.training.mask_classification_instance
import eomt.datasets.cityscapes_semantic
import eomt.datasets.lightning_data_module
from eomt.datasets.anomaly_dataset import AnomalyDataModule

def main():
    """
    Entry point per il training tramite PyTorch Lightning CLI.
    """
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        subclass_mode_model=True,
        subclass_mode_data=True,
    )

if __name__ == '__main__':
    main()
