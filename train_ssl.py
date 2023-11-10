from pathlib import Path

import torch
from tqdm import tqdm

from data import get_dataloaders, get_dataset_ssl
from model import ResNetPlus

# TODO: better config? Maybe not necessary
LEARNING_RATE = 5e-2
EPOCHS = 30
WEIGHTS_FOLDER = Path("./weights")
IMAGES_PATH = "./.data/raw-img"
BATCH_SIZE = 32
LOAD_WEIGHTS = True
RESNET_WEIGHTS="weights_ssl_cont.pt"
SAVE_NAME="weights_ssl_cont_2.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    WEIGHTS_FOLDER.mkdir(exist_ok=True)

    print(f"Running on device: {device}")

    model: torch.nn.Module = ResNetPlus().to(device)
    if LOAD_WEIGHTS:
        model.load_resnet((WEIGHTS_FOLDER / RESNET_WEIGHTS))
    optimizer = torch.optim.SGD(
        model.parameters(),
        LEARNING_RATE * BATCH_SIZE / 256,
        weight_decay=1e-4,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS
    )
    dataset = get_dataset_ssl(IMAGES_PATH)
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(
        dataset, batch_size=BATCH_SIZE
    )
    ssl_loss_criterion = torch.nn.CosineSimilarity()
    for epoch in range(EPOCHS):
        total_loss = 0
        print(f"Epoch {epoch + 1}")
        for index, batch in (
            pbar := tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        ):
            x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            z1, z2 = model.f(x1), model.f(x2)  # projections, n-by-d
            p1, p2 = model.predictor(z1), model.predictor(z2)  # predictions, n-by-d

            z1_no_grad, z2_no_grad = z1.detach(), z2.detach()
            loss = -(
                (
                    ssl_loss_criterion(z1_no_grad, p2)
                    + ssl_loss_criterion(z2_no_grad, p1)
                )
                / 2
            ).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().cpu()
            pbar.set_postfix_str(
                f"average loss {total_loss/(index+1):.3f}, batch std {torch.nn.functional.normalize(z1, dim=1).std(dim=0).mean():.4f}"
            )
        scheduler.step()
        model.save_resnet((WEIGHTS_FOLDER / SAVE_NAME).as_posix())


if __name__ == "__main__":
    main()
