import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=True)

        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.product = nn.Linear(256,9)
        self.global_head = nn.Linear(256,4)
        self.visual = nn.Sequential(
            nn.Linear(256,4),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.backbone(x).squeeze()

        x = self.fc(x)

        return self.product(x), self.global_head(x), self.visual(x)