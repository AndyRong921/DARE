import torch
import torch.nn as nn
import numpy as np
import joblib

class PCA_PDBL_Head(nn.Module):
    def __init__(self, pca_path, pdbl_weight_path):
        super().__init__()
        self.pca = joblib.load(pca_path)
        self.W = np.load(pdbl_weight_path)  

    def forward(self, feats):  
        feats_np = feats.detach().cpu().numpy()
        feats_pca = self.pca.transform(feats_np)
        logits = feats_pca @ self.W
        return torch.from_numpy(logits).float().to(feats.device)