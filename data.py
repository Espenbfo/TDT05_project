from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transforms: Callable):
        self.dataset = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.dataset[index]
        x1, x2 = self.transforms(image), self.transforms(image)
        return x1, x2


def get_dataset_ssl(path: str):
    parent_folder = Path(path).resolve().as_posix()
    image_dataset = ImageFolder(parent_folder)
    trans = _get_transforms()
    return ImageDataset(image_dataset, trans)


def get_dataset_supervised(path: str):
    parent_folder = Path(path).resolve().as_posix()
    trans = _get_transforms_supervised()

    return ImageFolder(parent_folder, trans)


def get_dataloaders(dataset: Dataset, batch_size=32, shuffle=True, num_workers=4):
    train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_dl = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl


def _get_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.GaussianBlur(5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
        ]
    )


def _get_transforms_supervised():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
        ]
    )
