from pathlib import Path

import torch
from tqdm import tqdm

from data import get_dataloaders, get_dataset
from model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 1e-4
EPOCHS = 10
WEIGHTS_FOLDER = Path("./weights")
IMAGES_PATH = "./.data/raw-img"


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    WEIGHTS_FOLDER.mkdir(exist_ok=True)

    print(f"Running on device: {device}")

    model: torch.nn.Module = ResNetPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    dataset = get_dataset(IMAGES_PATH)
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(dataset)
    ssl_loss_criterion = torch.nn.CosineSimilarity()
    for epoch in range(EPOCHS):
        total_loss = 0
        print(f"Epoch {epoch + 1}")
        for index, batch in (pbar := tqdm(enumerate(dataloader_train))):
            x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            z1, z2 = model.resnet(x1), model.resnet(x2)  # projections, n-by-d
            p1, p2 = model.predictor(z1), model.predictor(z2)  # predictions, n-by-d

            z1_no_grad, z2_no_grad = z1.detach(), z2.detach()
            loss = -(ssl_loss_criterion(z1_no_grad, p2) + ssl_loss_criterion(
                z2_no_grad, p1
            )).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            pbar.set_postfix_str(
                f"current average loss this epoch {total_loss/(index+1)}"
            )
    model.save_resnet((WEIGHTS_FOLDER / "weights.pt").as_posix())


if __name__ == "__main__":
    main()
