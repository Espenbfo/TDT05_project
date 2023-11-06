from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Animals10:
    def __init__(self, path: str) -> None:
        self.parent_folder: str = Path(path).resolve().as_posix()
        self.dataset = ImageFolder(self.parent_folder)

    def get_dataset(self):
        return self.dataset

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        return DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


class Transforms:
    transform_list = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
        ]
    )
