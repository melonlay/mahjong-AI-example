"""
定義用於麻將牌分類的簡單卷積神經網絡 (CNN) 模型。

此模組包含 `SimpleMahjongCNN` 類，繼承自 `torch.nn.Module`。
模型結構包括：
- 三個卷積層 (Conv2d)，每個後面跟著批次標準化 (BatchNorm2d)、ReLU 激活函數和最大池化層 (MaxPool2d)。
- 兩個全連接層 (Linear)，中間有一個 Dropout 層用於正則化。
- 自動計算卷積層輸出到第一個全連接層輸入的特徵數量 (`_determine_fc_input_size`)。

用法:
由訓練 (`train.py`)、評估 (`eval.py`) 和推論 (`infer_and_organize.py`) 腳本導入和實例化。

- 在訓練/評估/推論腳本中:
  ```python
  from trainer.classification.model import SimpleMahjongCNN

  num_classes = 30 # 假設包含 noise 共 30 類
  input_size = 96 # 假設輸入圖片尺寸為 96x96
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = SimpleMahjongCNN(num_classes=num_classes, input_size=input_size).to(device)

  # 加載數據和權重...
  # output = model(input_batch)
  ```
也可以直接運行此文件來打印模型結構和參數信息:
```bash
python trainer/classification/model.py
```
"""
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
