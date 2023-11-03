from typing import Callable

import torch
from torch.utils.data import DataLoader

from .data import Animals10
from .data import Transforms
from .model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 1e-3
EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
data = Animals10(".data/raw-img")


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
    model: torch.nn.Module = get_model()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    dataloader = get_dataloader()
    z1, z2 = model.resnet(x1), model.resnet(x2)  # projections, n-by-d
    p1, p2 = model.predictor(z1), model.predictor(z2)  # predictions, n-by-d
    z1_no_grad, z2_no_grad = z1.detach(), z2.detach()
    loss = ssl_loss_criterion(z1_no_grad, p2) + ssl_loss_criterion(
    z2_no_grad, p1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    


if __name__ == "__main__":
    main()
