import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(EyeClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28
        )
        # Тепер флеттен розмір: 128 × 28 × 28 = 100352
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

if __name__ == "__main__":
    # Для тесту
    model = EyeClassifierCNN(num_classes=4)
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # має бути [1, 4]
