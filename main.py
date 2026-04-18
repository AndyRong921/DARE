import os
import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
import joblib


from tool.dataset import ImageDataset
from tool.PDBL import PDBL_net
import timm
from pdbl_swin_tiny_model import SwinTripleBackbone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name):
    if name == "resnet50":
        base_model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    elif name == "eff":
        base_model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    elif name == "swin":
        return SwinTripleBackbone(backbone_name="swin_tiny_patch4_window7_224").to(device)
    elif name == "shuffle":
        base_model = timm.create_model("shufflenet_v2_x1_0", pretrained=True, num_classes=0)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    class TripleBackbone(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, img224, img160, img112):
            f1 = self.backbone(img224)
            f2 = self.backbone(img160)
            f3 = self.backbone(img112)
            return torch.cat([f1, f2, f3], dim=1)

    return TripleBackbone(base_model).to(device)



def split_dataset(dataset, n_split):
    indices = np.random.permutation(len(dataset))
    size = len(dataset) // n_split
    return [Subset(dataset, indices[i * size:(i + 1) * size]) for i in range(n_split)]



def compute_dare_stats(dataset, model=None, mode="DARE*", batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    means, stds = [], []
    
    if mode == "DARE*":
        for img224, img160, img112, _ in tqdm(loader, desc="[Stats] DARE*"):
            for img in [img224, img160, img112]:
                img = img.float() / 255.0
                means.append(img.mean(dim=[0, 2, 3]))
                stds.append(img.std(dim=[0, 2, 3]) + 1e-6)
    else:
        model.eval()
        with torch.no_grad():
            for img224, img160, img112, _ in tqdm(loader, desc="[Stats] DARE+"):
                feat = model(img224.to(device), img160.to(device), img112.to(device))
                means.append(feat.mean(dim=0).cpu())
                stds.append(feat.std(dim=0).cpu() + 1e-6)
                
    return torch.stack(means).mean(0), torch.stack(stds).mean(0)

def apply_dare(data, means, stds):
    mu, sigma = random.choice(list(zip(means, stds)))
    if len(data.shape) == 4: # Image
        return (data - mu[None, :, None, None].to(data.device)) / sigma[None, :, None, None].to(data.device)
    else: # Feature
        return (data - mu.to(data.device)) / (sigma.to(data.device) + 1e-6)


def extract_features_dare_mode(model, loader, star_pool=None, plus_pool=None, mode="Baseline"):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for img224, img160, img112, target in tqdm(loader, desc=f"Extracting {mode}"):
            img224, img160, img112 = img224.float().to(device), img160.float().to(device), img112.float().to(device)

            if mode == "DARE*" and star_pool:
                img224 = apply_dare(img224, *star_pool)
                img160 = apply_dare(img160, *star_pool)
                img112 = apply_dare(img112, *star_pool)

            feat = model(img224, img160, img112)

            if mode == "DARE+" and plus_pool:
                feat = apply_dare(feat, *plus_pool)

            all_feats.append(feat.cpu().numpy())
            all_labels.append(target.argmax(dim=1).cpu().numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)


def run_dare_ablation(args):
    source_train = ImageDataset(args.source_dir, n_class=args.n_class)
    target_test = ImageDataset(args.target_dir, n_class=args.n_class)
    target_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.backbone)
    pca = joblib.load(f"{args.save_dir}/PCA_{args.backbone}.pkl")


    source_clients = split_dataset(source_train, args.n_clients)
    star_pool = ([], [])
    plus_pool = ([], [])
    for client in source_clients:
        ms, ss = compute_dare_stats(client, mode="DARE*")
        star_pool[0].append(ms); star_pool[1].append(ss)
        mp, sp = compute_dare_stats(client, model=model, mode="DARE+")
        plus_pool[0].append(mp); plus_pool[1].append(sp)

 
    f_b, y_true = extract_features_dare_mode(model, target_loader, mode="Baseline")
    pdbl_b = PDBL_net(isPCA=False, n_components=f_b.shape[1], reg=0.001)
    pdbl_b.train(pca.transform(f_b), np.eye(args.n_class)[y_true])
    p_b = pdbl_b.predict(pca.transform(f_b))

 
    f_s, _ = extract_features_dare_mode(model, target_loader, star_pool=star_pool, mode="DARE*")
    pdbl_s = PDBL_net(isPCA=False, n_components=f_s.shape[1], reg=0.001)
    pdbl_s.train(pca.transform(f_s), np.eye(args.n_class)[y_true])
    p_s = pdbl_s.predict(pca.transform(f_s))


    f_p, _ = extract_features_dare_mode(model, target_loader, plus_pool=plus_pool, mode="DARE+")
    pdbl_p = PDBL_net(isPCA=False, n_components=f_p.shape[1], reg=0.001)
    pdbl_p.train(pca.transform(f_p), np.eye(args.n_class)[y_true])
    p_p = pdbl_p.predict(pca.transform(f_p))


    p_vote = (p_b + p_s + p_p) / 3.0
    
    def get_metrics(y, prob):
        preds = prob.argmax(axis=1)
        return [accuracy_score(y, preds), f1_score(y, preds, average='macro'), matthews_corrcoef(y, preds)]

    res_b = get_metrics(y_true, p_b)
    res_s = get_metrics(y_true, p_s)
    res_p = get_metrics(y_true, p_p)
    res_v = get_metrics(y_true, p_vote)

    print("\n" + "="*50)
    print(f"Backbone: {args.backbone}")
    print(f"Method    | Acc    | F1     | MCC")
    print(f"Baseline  | {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f}")
    print(f"DARE* | {res_s[0]:.4f} | {res_s[1]:.4f} | {res_s[2]:.4f}")
    print(f"DARE+     | {res_p[0]:.4f} | {res_p[1]:.4f} | {res_p[2]:.4f}")
    print(f"Voting    | {res_v[0]:.4f} | {res_v[1]:.4f} | {res_v[2]:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="swin", choices=["resnet50", "eff", "swin", "shuffle"])
    parser.add_argument("--source_dir", default="dataset/KME")
    parser.add_argument("--target_dir", default="dataset/Kather")
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_class", default=9, type=int)
    parser.add_argument("--n_clients", default=4, type=int)
    args = parser.parse_args()
    
    run_dare_ablation(args)