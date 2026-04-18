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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = timm.create_model("shufflenet_v2_x1_0", pretrained=True, num_classes=0)
    return model.to(device)

def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for img224, img160, img112, label in tqdm(dataloader, desc="Extracting features"):
            img224, img160, img112 = img224.to(device), img160.to(device), img112.to(device)
            label = label.to(device)
            feats = []
            for img in [img224, img160, img112]:
                f = model(img)
                feats.append(f)
            feat_cat = torch.cat(feats, dim=1)
            features.append(feat_cat.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.vstack(features), np.concatenate(labels)

def triple_resize_collate(batch):
    imgs, labels = zip(*batch)
    img224 = torch.stack([transforms.Resize((224, 224))(img) for img in imgs])
    img160 = torch.stack([transforms.Resize((160, 160))(img) for img in imgs])
    img112 = torch.stack([transforms.Resize((112, 112))(img) for img in imgs])
    labels = torch.tensor(labels)
    return img224, img160, img112, labels

def prepare_dataloader(source_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=source_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                      collate_fn=triple_resize_collate)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n>> Loading ShuffleNetV2 model")
    model = build_model()

    print(f">> Loading image dataset: {args.source_dir}")
    dataloader = prepare_dataloader(args.source_dir, args.batch_size)

    print(">> Extracting image features...")
    features, labels = extract_features(model, dataloader)
    print(f"Feature dimensions: {features.shape}")

    print(">> Training PCA model and saving...")
    pca = PCA(n_components=256)
    features_pca = pca.fit_transform(features)
    pca_path = os.path.join(args.save_dir, "PCA_shuff.pkl")
    joblib.dump(pca, pca_path)

    print(">> Training PDBL classifier and saving...")
    pdbl = PDBL_net(isPCA=False, n_components=256, reg=0.001)
    pdbl.train(features_pca, np.eye(args.n_class)[labels])
    pdbl_path = os.path.join(args.save_dir, "PDBL_shuff.npy")
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