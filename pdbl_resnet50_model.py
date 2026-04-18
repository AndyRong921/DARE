import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tool.resnet import resnet50
from tool.pdbl_head import PCA_PDBL_Head 

class ResNet50WithPDBL(nn.Module):
    def __init__(self, pca_path, pdbl_path):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.head = PCA_PDBL_Head(pca_path, pdbl_path)

    def forward(self, x):
        feats = self.backbone(x) 
        logits = self.head(feats)
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"\n Accuracy: {accuracy:.2f}%")

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = ResNet50WithPDBL(
        pca_path='./save/PCA_r50.pkl',
        pdbl_path='./save/PDBL_r50.npy'
    ).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset = Subset(dataset, range(2000))  
    dataloader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)

    evaluate_model(model, dataloader, device)

if __name__ == '__main__':
    main()