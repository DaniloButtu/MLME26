# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/MLME26/trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'max_entropy'])
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    all_logits_to_save = []
    all_gts_to_save = []
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        # anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)   
        if args.method == "msp":
            # Maximum Softmax Probability
            probs = torch.softmax(result, dim=1)
            msp, _ = torch.max(probs.squeeze(0), dim=0)
            anomaly_result = 1.0 - msp.data.cpu().numpy()

        elif args.method == "max_logit":
            # Max Logit: il valore massimo dei logits (negato per avere score alto = anomalia)
            anomaly_result = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)

        elif args.method == "max_entropy":
            # Max Entropy: incertezza basata sulla distribuzione Shannon
            probs = torch.softmax(result, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            anomaly_result = entropy.squeeze(0).data.cpu().numpy()

        ######################################################################################################################
        # SALVATAGGIO IMMAGINE DI DEBUG (prima immagine utile)
        if len(ood_gts_list) == 0: 
            print(f"Debug: {path}")
            map_normalized = cv2.normalize(anomaly_result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_img = cv2.applyColorMap(map_normalized, cv2.COLORMAP_JET)
            debug_name = "debug_heatmap.jpg"
            cv2.imwrite(debug_name, heatmap_img)
            print(f"\n--- MAPPA DI DEBUG SALVATA COME {debug_name} ---")
            print("Guardala per capire cosa la rete sta marcando come anomalia!\n")
        ################################################################################################################

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
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
        
        all_logits_to_save.append(result.squeeze(0).cpu().numpy())
        all_gts_to_save.append(ood_gts.astype(np.uint8))
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    valid_mask = (ood_gts == 0) | (ood_gts == 1)
    val_out = anomaly_scores[valid_mask]
    val_label = ood_gts[valid_mask]

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()
    
    np.save("logits_dump.npy", np.array(all_logits_to_save))
    np.save("gts_dump.npy", np.array(all_gts_to_save))
    print("Salvataggio completato! File per la Temperature Scaling pronti.")

if __name__ == '__main__':
    main()
