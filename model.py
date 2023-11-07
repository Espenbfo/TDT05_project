import torch
import torch.nn as nn
from torchvision.models import resnet50

# Backbone: ResNet50
# Predictor head:
# Classification head: FC layer


class ResNetPlus(nn.Module):
    def __init__(self, output_dim: int = 2048, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = resnet50(weights=None, **kwargs)
        self.resnet.fc = torch.nn.Identity()
        self.f = nn.Sequential(self.resnet,
                               nn.Linear(2048, 2048),
                               nn.ReLU(),
                               nn.BatchNorm1d(2048),
                               nn.Linear(2048, 2048),
                               nn.ReLU(),
                               nn.BatchNorm1d(2048),
                               nn.Linear(2048, 2048))
        
        self.predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
        )

    def save_resnet(self, path="resnet-50.pt"):
        torch.save(self.resnet, path)

    def forward(self, x_1, x_2, use_predictor: bool = True):
        x_1 = self.f(x_1)
        if use_predictor is not None:
            x_1 = self.predictor(x_1)
        x_2 = self.f(x_2)
        return x_1, x_2


def main():
    model = ResNetPlus()
    x = torch.randn(1, 3, 244, 244)
    y = torch.randn(1, 3, 244, 244)
    z = model(x, y)
    print(z)


if __name__ == "__main__":
    main()
