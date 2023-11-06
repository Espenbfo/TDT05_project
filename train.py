from typing import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import Animals10, Transforms
from model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 1e-4
EPOCHS = 10
WEIGHTS_FOLDER = Path("weights")
DATA_FOLDER = Path("E:\\datasets\\animal10")


device = "cuda" if torch.cuda.is_available() else "cpu"
data = Animals10(DATA_FOLDER)


# Replace with import from model.py
def get_model():
    return ResNetPlus(output_dim=1000)


# Replace with import from data.py
def get_dataloader():
    return data.get_dataloader()


# Replace with import from data.py
def get_transforms():
    return Transforms.transform_list


def main():
    WEIGHTS_FOLDER.mkdir(exist_ok=True)

    print(f"Running on device: {device}")
    model: torch.nn.Module = get_model()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    dataset = get_dataset()
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders()
    ssl_loss_criterion = torch.nn.CosineSimilarity()
    for epoch in range(EPOCHS):
        total_loss = 0
        print(f"Epoch {epoch + 1}")
        for index, batch in (pbar := enumerate(dataloader_train)):
            x1, x2 = batch.to(device)

            z1, z2 = model.resnet(x1), model.resnet(x2)  # projections, n-by-d
            p1, p2 = model.predictor(z1), model.predictor(z2)  # predictions, n-by-d

            z1_no_grad, z2_no_grad = z1.detach(), z2.detach()
            loss = ssl_loss_criterion(z1_no_grad, p2) + ssl_loss_criterion(
                z2_no_grad, p1
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            pbar.suffix = f"current total loss this epoch {total_loss}"
    model.save_resnet(WEIGHTS_FOLDER/"weights.pt")


if __name__ == "__main__":
    main()
