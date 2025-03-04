import torch
import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # サブディレクトリを再帰的に探索し、すべての画像ファイルのパスを収集
        assert os.path.exists(root_dir) == True
        for subdir, _, files in os.walk(root_dir):
            # print("files", files)
            for file in files:
                if file.endswith(".png"):  # 必要に応じて他の拡張子もチェック
                    self.image_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataset(dataset_key):
    # CIFAR10の読み込み
    if dataset_key == "cifar10":
        dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    
    elif dataset_key == "ffhq32":
        dataset = FFHQDataset(
            root_dir="./data/ffhq_dataset/ffhq32",            
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    elif dataset_key == "ffhq64":
        dataset = FFHQDataset(
            root_dir="./data/ffhq_dataset/ffhq64",            
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    else:
        dataset = None   
    
    return dataset
