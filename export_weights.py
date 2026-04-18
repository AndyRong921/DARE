# export_weights.py
import pickle
import numpy as np
from sklearn.decomposition import PCA
with open('./save/PDBL_on_r50.pkl', 'rb') as f:
    pdbl_model = pickle.load(f)
W = pdbl_model.W 
pca = PCA(n_components=900)
W_reduced = pca.fit_transform(W)  
np.save('./save/pdbl_r50_900.npy', W_reduced)
print("succeed to the  pdbl_r50_900.npy")