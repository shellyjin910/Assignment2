# model.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    题面 CNN 结构（输入 64x64x3）：
    Conv2d(3→16, k3, s1, p1) → ReLU → MaxPool(2,2)
    Conv2d(16→32, k3, s1, p1) → ReLU → MaxPool(2,2)
    Flatten → FC(8192→100) → ReLU → FC(100→10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 64x64 → 池化两次后变 16x16，通道 32 → 32*16*16=8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # quick shape test
    m = SimpleCNN()
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 10)
    print("OK:", y.shape)
