from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataloaders, get_dataset_supervised
from model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 1e-4
EPOCHS = 500
WEIGHTS_FOLDER = Path("./weights")
IMAGES_PATH = ".data_sv/"
BATCH_SIZE = 64
FREEZE_BACKBONE = True
LOAD_BACKBONE_WEIGHTS = True
SAVE_NAME="weights_supervised.pt"
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
        for param in model.resnet.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        LEARNING_RATE,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, LEARNING_RATE*BATCH_SIZE/(256*100))
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(
        dataset, batch_size=BATCH_SIZE
    )
    loss_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        best_loss = 1000000
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
        val_loss = test_val(model, dataloader_val, loss_criterion, "Validation")
        if val_loss <= best_loss:
            best_loss = val_loss
            model.save_resnet((WEIGHTS_FOLDER / SAVE_NAME).as_posix())
        print("\n")
    test_val(model, dataloader_test, loss_criterion, "Test")


def test_val(
    model: nn.Module,
    dataloader_val: DataLoader,
    loss_criterion: Callable,
    testing_type: str = "Validation",
):
    print(f"{testing_type}")
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
    return total_loss/len(dataloader_val)


if __name__ == "__main__":
    main()
