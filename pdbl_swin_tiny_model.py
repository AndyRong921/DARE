import os
import argparse
import joblib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

from tool.PDBL import PDBL_net
from sklearn.decomposition import PCA
from torchvision.transforms import Resize, ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SwinTripleBackbone(nn.Module):
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        self.backbone1 = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone3 = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.out_dim = {
            "swin_tiny_patch4_window7_224": 768,
            "swin_base_patch4_window7_224": 1024
        }[backbone_name]

    def forward(self, img224, img160, img112):
        f1 = self.backbone1(img224)  
        f2 = self.backbone2(img160)
        f3 = self.backbone3(img112)
        return torch.cat([f1, f2, f3], dim=1)  


def build_model():
    model = SwinTripleBackbone(backbone_name="swin_tiny_patch4_window7_224")
    return model.to(device)


def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        # Extract features
        for img224, img160, img112, label in tqdm(dataloader, desc="Extracting features"):
            img224, img160, img112 = img224.to(device), img160.to(device), img112.to(device)
            label = label.to(device)
            feat_cat = model(img224, img160, img112)  
            features.append(feat_cat.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.vstack(features), np.concatenate(labels)


def triple_resize_collate(batch):
    imgs, labels = zip(*batch)

    resize_to_224 = Resize((224, 224))
    to_tensor = ToTensor()

    def process(img, size):
        return to_tensor(resize_to_224(Resize(size)(img)))  

    img224 = torch.stack([process(img, 224) for img in imgs])
    img160 = torch.stack([process(img, 160) for img in imgs])
    img112 = torch.stack([process(img, 112) for img in imgs])
    labels = torch.tensor(labels)
    return img224, img160, img112, labels


def prepare_dataloader(source_dir, batch_size):
    dataset = datasets.ImageFolder(root=source_dir)  
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                      collate_fn=triple_resize_collate)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n>> Loading Swin-Tiny triple-branch model")
    model = build_model()

    print(f">> Loading image dataset: {args.source_dir}")
    dataloader = prepare_dataloader(args.source_dir, args.batch_size)

    print(">> Extracting image features...")
    features, labels = extract_features(model, dataloader)
    print(f"Feature dimension: {features.shape}")

    print(">> Training and saving PCA model...")
    pca = PCA(n_components=256)
    features_pca = pca.fit_transform(features)
    pca_path = os.path.join(args.save_dir, "PCA_swin.pkl")
    joblib.dump(pca, pca_path)

    print(">> Training and saving PDBL classifier...")
    pdbl = PDBL_net(isPCA=False, n_components=256, reg=0.001)
    pdbl.train(features_pca, np.eye(args.n_class)[labels])
    pdbl_path = os.path.join(args.save_dir, "PDBL_swin.npy")
    np.save(pdbl_path, pdbl.W)

    print(f"\n Saved: \n- {pca_path}\n- {pdbl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="dataset/KME")
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_class", default=9, type=int)
    args = parser.parse_args()
    main(args)