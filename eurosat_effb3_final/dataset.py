import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchgeo.datasets as geo_datasets

import config

def process_patch(sample, clip_val=3000):
    """
    Convert raw Sentinel-2 satellite image tensor to 8-bit RGB PIL Image.
    Bands B4 (Red), B3 (Green), B2 (Blue) have reflectance values up to 10000+.
    Clipping at 3000 capture the most useful dynamic range.
    """
    # permute (C, H, W) -> (H, W, C)
    img_np = sample['image'].permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, clip_val)
    img_np = (img_np / clip_val * 255).astype(np.uint8)
    return Image.fromarray(img_np)

class EuroSATCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_and_split_data():
    """
    Downloads (if not cached) and loads the full EuroSAT optical dataset,
    converts tensors to PIL, and performs stratified 70/15/15 splits.
    """
    print(f"Loading EuroSAT dataset from '{config.DATA_DIR}'...")
    full_dataset = geo_datasets.EuroSAT(
        root=config.DATA_DIR,
        bands=['B04', 'B03', 'B02'],
        download=True
    )
    
    total_samples = len(full_dataset)
    print(f"   Total raw samples loaded: {total_samples:,}")
    print("Processing satellite bands into RGB images (clipping at 3000)...")
    
    images = []
    labels = []
    
    for idx in range(total_samples):
        sample = full_dataset[idx]
        images.append(process_patch(sample))
        labels.append(sample['label'].item())
        
        if (idx + 1) % 5000 == 0 or (idx + 1) == total_samples:
            print(f"   Processed: {idx + 1:,} / {total_samples:,} patches")
            
    # Perform stratified splits (70% Train, 15% Val, 15% Test)
    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        images, labels, test_size=0.3, random_state=config.SEED, stratify=labels
    )
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        test_imgs, test_lbls, test_size=0.5, random_state=config.SEED, stratify=test_lbls
    )
    
    print(f"Split complete:")
    print(f"   Train samples: {len(train_imgs):,}")
    print(f"   Val samples  : {len(val_imgs):,}")
    print(f"   Test samples : {len(test_imgs):,}")
    
    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)

def get_transforms(img_size):
    """
    Defines data augmentations for training and simple transforms for validation/inference.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_transform

def get_dataloaders(train_data, val_data, test_data, img_size, batch_size=32):
    """
    Builds and returns PyTorch DataLoader objects for train, val, and test sets.
    """
    train_imgs, train_lbls = train_data
    val_imgs, val_lbls = val_data
    test_imgs, test_lbls = test_data
    
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = EuroSATCustomDataset(train_imgs, train_lbls, transform=train_transform)
    val_dataset = EuroSATCustomDataset(val_imgs, val_lbls, transform=val_transform)
    test_dataset = EuroSATCustomDataset(test_imgs, test_lbls, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
