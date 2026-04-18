import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):

    def __init__(self, data_path, n_class):
        self._data_path = data_path
        self.n_class = n_class
        self._normalize = True
 
        self._resize224 = transforms.Resize((224, 224))  
        self._resize160 = transforms.Compose([
            transforms.Resize((160, 160)),               
            transforms.Resize((224, 224))                
        ])
        self._resize112 = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Resize((224, 224))
        ])

        self._pre_process()

    def _pre_process(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self._items = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    ext = fname.split('.')[-1].lower()
                    if ext in ['tif', 'jpeg', 'png', 'jpg']:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)
        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        label = np.array(label, dtype=float)

        img = Image.open(path).convert('RGB')

  
        img1 = self._resize224(img)
        img2 = self._resize160(img)
        img3 = self._resize112(img)

  
        def preprocess(pil_img):
            arr = np.array(pil_img, dtype=np.float32).transpose((2, 0, 1))  
            return (arr - 128.0) / 128.0 if self._normalize else arr

        img1 = preprocess(img1)
        img2 = preprocess(img2)
        img3 = preprocess(img3)

        label_onehot = np.zeros((self.n_class), dtype=np.float32)
        label_onehot[int(label)] = 1.0

        return img1, img2, img3, label_onehot


if __name__ == "__main__":
    dataset = ImageDataset(data_path="dataset/KME", n_class=9)
    print("sample number:", len(dataset))
    img1, img2, img3, label = dataset[0]
    print("img224 shape:", img1.shape)
    print("img160 shape:", img2.shape)
    print("img112 shape:", img3.shape)
    print("label onehot:", label)