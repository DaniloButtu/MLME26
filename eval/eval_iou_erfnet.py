import os
import zipfile
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor
from erfnet import ERFNet          # import the ERFNet model
from iouEval import iouEval, getColorEntry


# Define the number of classes for IoU.
# NUM_CLASSES = 20 because the metric considers 19 real classes (trainId 0-18) plus void at index 19.
# The ERFNet model is trained on 20 output channels, the last one is ignored in IoU calculation.
NUM_CLASSES = 20

##############################################################################
# Mapping from original Cityscapes labelId (0-255) to trainId (0-19).
# Create an array of 256 elements, all initialized to 19 (void).
# Then overwrite only the positions corresponding to the 19 classes of interest.
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
# Custom dataset to read images and labels directly from ZIP files,
# without extracting them. Supports splits 'val', 'train', 'test', etc.
# Unlike the EoMT version, here images and labels are forcibly resized to 512×1024 (standard format for ERFNet on Cityscapes).
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

        # Read the label bytes from the ZIP and convert to a PIL image (grayscale)
        lbl_bytes = self.lbl_zip.read(lbl_name)
        label = Image.open(BytesIO(lbl_bytes))

        # Resize to 512x1024 (standard format for ERFNet on Cityscapes)
        image = resize(image, (512, 1024), interpolation=Image.BILINEAR)
        label = resize(label, (512, 1024), interpolation=Image.NEAREST)

        # RECALL: to_tensor converts the image to a float32 tensor and automatically divides pixels by 255 (scales to [0,1])
        image = to_tensor(image)                     # [3, 512, 1024] float32

        # Convert label to uint8 numpy array, apply labelId -> trainId mapping,
        # convert to torch.long tensor and add a channel dimension (1, H, W)
        label = np.array(label, dtype=np.uint8)
        label = label2train[label]
        label = torch.from_numpy(label).long().unsqueeze(0)  # [1, 512, 1024]

        return image, label

def main():
    # Command line argument parsing
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default='/content/MLME26/trained_models/',
                        help='Directory where the weights are located')
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth',
                        help='Name of the weights file (e.g. erfnet_pretrained.pth)')
    parser.add_argument('--img-zip', required=True,
                        help='Path to the ZIP file of images (leftImg8bit_trainvaltest.zip)')
    parser.add_argument('--lbl-zip', required=True,
                        help='Path to the ZIP file of labels (gtFine_trainvaltest.zip)')
    parser.add_argument('--subset', default='val',
                        help='Split to evaluate: val, train, test')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of processes for the DataLoader')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (ERFNet processes whole images, so can be >1)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force execution on CPU even if CUDA is available')
    args = parser.parse_args()

    # Determine the device (CUDA if available and not explicitly requested CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    #########################################################
    # Instantiate the ERFNet model with 20 classes
    ##############################################
    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    # Build the full path to the weights file
    weightspath = os.path.join(args.loadDir, args.loadWeights)
    # Load the state_dict
    state_dict = torch.load(weightspath, map_location='cpu')

    # Function to load the state_dict by removing any 'module.' prefix present in checkpoints
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # If the name starts with 'module.', try to remove the prefix
                if name.startswith("module."):
                    stripped_name = name.split("module.")[-1]
                    if stripped_name in own_state:
                        own_state[stripped_name].copy_(param)
                    else:
                        print(f"{stripped_name} not found in model")
                else:
                    print(f"{name} not loaded")
                continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, state_dict)
    model.eval()   # Set evaluation mode
    print("ERFNet model and weights loaded successfully")

    ###########################################################
    # Create DataLoader using the ZIP-based dataset
    dataset = ZipCityscapesDataset(args.img_zip, args.lbl_zip, split=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Instance of the class that computes IoU (20 classes, ignore index 19)
    iou_eval = iouEval(NUM_CLASSES)

    # Evaluation loop
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            # Move images and labels to the device
            if not args.cpu:
                images = images.cuda()
                labels = labels.cuda()

            # Forward of the whole image (size 512x1024)
            outputs = model(images)                     # shape [B, 20, 512, 1024]
            # Extract the class with maximum probability along the channel dimension (dim=1)
            # and add a channel dimension (keepdim=False, then unsqueeze)
            preds = outputs.max(dim=1)[1].unsqueeze(1)   # [B, 1, 512, 1024]

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