import torch
import torch.nn as nn, optim
from torch.utils.data import DataLoader, random_split
from torch.vision import transforms, datasets
from model import SimpleCNN
from ImageDataSet import ImageDataSet
import argparse, os, random

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loaders(data_root, batch_size, num_workers, val_split=5000, augment=True, seed=42):
    """Split the dataset into training and validation sets and create DataLoaders."""
    tf_train = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if augment:
        tf_train.append(transforms.RandomErasing(p = 0.5))

    train_tfms = transforms.Compose(tf_train)
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train =datasets.CIFAR10(root= data_root, train=True, download=True, transform=train_tfms) 
    test_ds =datasets.CIFAR10(root= data_root, train=False, download=True, transform=train_tfms) 

    g = torch.Generator().manual_seed(seed)
    train_len = len(full_train) - val_split
    train_ds, val_ds = random_split(full_train, [train_len, val_split], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += inputs.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
    