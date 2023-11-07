from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataloaders, get_dataset_supervised
from model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 5e-3
EPOCHS = 50
WEIGHTS_FOLDER = Path("./weights")
IMAGES_PATH = "../catmatch/.data/"
BATCH_SIZE = 32
FREEZE_BACKBONE = False
LOAD_BACKBONE_WEIGHTS = False
BACKBONE_WEIGHT_PATH = Path(WEIGHTS_FOLDER / "weights_ssl.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    WEIGHTS_FOLDER.mkdir(exist_ok=True)

    print(f"Running on device: {device}")

    dataset = get_dataset_supervised(IMAGES_PATH)

    classes = dataset.classes
    model: torch.nn.Module = ResNetPlus(len(classes), is_classifier=True).to(device)

    if LOAD_BACKBONE_WEIGHTS:
        model.load_resnet(BACKBONE_WEIGHT_PATH)
    if FREEZE_BACKBONE:
        model.resnet.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        LEARNING_RATE * BATCH_SIZE / 256,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, LEARNING_RATE*BATCH_SIZE/(256*100))
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(
        dataset, batch_size=BATCH_SIZE
    )
    loss_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        print(f"Epoch {epoch + 1}")
        for index, batch in (
            pbar := tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        ):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
            loss = loss_criterion(pred, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            total_acc += (
                torch.sum(torch.eq(torch.argmax(pred, dim=1), labels)) / x.shape[0]
            )
            pbar.set_postfix_str(
                f"average loss {total_loss/(index+1):.3f}, average accuracy {total_acc/(index+1):.3f}"
            )
        test_val(model, dataloader_val, loss_criterion, "Validation")
        model.save_resnet((WEIGHTS_FOLDER / "weights_supervised.pt").as_posix())
    test_val(model, dataloader_test, loss_criterion, "Test")


def test_val(
    model: nn.Module,
    dataloader_val: DataLoader,
    loss_criterion: Callable,
    testing_type: str = "Validation",
):
    print(f"\n{testing_type}")
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for index, batch in (
            pbar := tqdm(enumerate(dataloader_val), total=len(dataloader_val))
        ):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
            loss = loss_criterion(pred, labels)
            loss.detach().cpu()
            total_loss += loss
            total_acc += (
                torch.sum(torch.eq(torch.argmax(pred, dim=1), labels)) / x.shape[0]
            )
            pbar.set_postfix_str(
                f"average loss {total_loss/(index+1):.3f}, average acc {total_acc/(index+1):.3f}"
            )


if __name__ == "__main__":
    main()
