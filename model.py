import torch
import torch.nn as nn
from torchvision.models import resnet50

# Backbone: ResNet50
# Predictor head:
# Classification head: FC layer


class ResNetPlus(nn.Module):
    def __init__(
        self, output_dim: int = 2048, is_classifier=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = resnet50(weights=None, **kwargs)
        self.resnet.fc = torch.nn.Identity()
        self.is_classifier = is_classifier
        if not self.is_classifier:
            self.f = nn.Sequential(
                self.resnet,
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Linear(2048, 2048),
            )

            self.predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, output_dim),
            )
        else:
            self.classifier_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, output_dim),
            )

    def save_resnet(self, path="resnet-50.pt"):
        torch.save(self.resnet.state_dict, path)

    def load_resnet(self, path):
        self.resnet.load_state_dict(path)

    def forward(self, x):
        assert self.is_classifier
        x = self.resnet(x)
        x = self.classifier_head(x)
        return x
            


def main():
    model = ResNetPlus()
    x = torch.randn(1, 3, 244, 244)
    y = torch.randn(1, 3, 244, 244)
    z = model(x, y)
    print(z)


if __name__ == "__main__":
    main()
