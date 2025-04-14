# trainer/classification/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMahjongCNN(nn.Module):
    def __init__(self, num_classes=29, input_size=96):
        super(SimpleMahjongCNN, self).__init__()

        # Simpler Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate FC input size dynamically
        self._determine_fc_input_size(input_size)

        # Simpler Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _determine_fc_input_size(self, input_size):
        # Use the defined conv layers for calculation
        # Create a dummy tensor with the expected input size
        # Ensure it's on the same device as the model might be moved later
        # Need to define layers before calling this
        dummy_input = torch.randn(1, 3, input_size, input_size)
        # Pass it through the convolutional and pooling layers defined above
        x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Calculate the number of features after flattening
        self.fc_input_features = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    input_dim = 96
    num_classes = 29
    model = SimpleMahjongCNN(num_classes=num_classes, input_size=input_dim)
    print(model)
    dummy_input = torch.randn(4, 3, input_dim, input_dim)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Calculated fc_input_features: {model.fc_input_features}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
