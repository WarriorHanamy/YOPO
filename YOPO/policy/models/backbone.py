import time

import torch
import torch.nn

from .resnet import resnet18


class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(
            512, output_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim):
    return ResNet18(output_dim)


if __name__ == '__main__':
    net = YopoBackbone(64)
    input_ = torch.zeros((1, 1, 96, 160))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
