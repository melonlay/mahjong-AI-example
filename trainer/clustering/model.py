"""
定義用於監督式對比學習 (Supervised Contrastive Learning, SupCon) 的
基於 ResNet 的特徵提取模型。

此模組包含 `SupConResNet` 類，繼承自 `torch.nn.Module`。
模型結構包括：
- 一個基於預訓練 ResNet18 的編碼器 (encoder)，移除其原始的分類頭 (`fc` 層)。
- 一個投影頭 (projection_head)，由兩個線性層和一個 ReLU 激活函數組成，
  將編碼器輸出的特徵映射到一個低維度的嵌入空間 (由 `embedding_dim` 控制)。

主要方法:
- `forward(x)`: 執行完整的前向傳播，通過編碼器和投影頭，
  最後返回 L2 歸一化 (normalized) 的嵌入向量。這些歸一化的嵌入向量
  主要用於在訓練階段計算 SupCon 損失。
- `get_features(x)`: 只執行編碼器部分的前向傳播，返回投影頭之前的原始特徵向量。
  此方法主要用於模型訓練完成後，在聚類或下游任務中提取圖像的特徵表示。

用法:
主要由聚類模型的訓練腳本 (`trainer/clustering/train.py`) 導入和實例化。
```python
from trainer.clustering.model import SupConResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupConResNet(embedding_dim=128).to(device)

# 在訓練循環中:
# (view1, view2), _ = batch
# inputs = torch.cat([view1, view2], dim=0).to(device)
# normalized_embeddings = model(inputs) # 用於計算 SupCon Loss

# 在推理/聚類前:
# features = model.get_features(single_image_batch) # 獲取用於聚類的特徵
```
此文件本身不能直接運行以產生功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import logging

logger = logging.getLogger(__name__)


class SupConResNet(nn.Module):
    """基於 ResNet 的模型，帶有投影頭，用於監督式對比學習"""

    def __init__(self, name='resnet18', embedding_dim=128):
        super(SupConResNet, self).__init__()
        if name == 'resnet18':
            # 嘗試使用較新的 weights API
            try:
                # 使用默認推薦的權重 (通常是 IMAGENET1K_V1)
                weights = ResNet18_Weights.DEFAULT
            except AttributeError:
                # 回退到舊的方式 (如果 torchvision 版本較舊)
                logger.warning(
                    "無法使用 ResNet18_Weights.DEFAULT，回退到 pretrained=True")
                base_model = resnet18(pretrained=True)
                weights = None  # 標記一下，避免下面重複加載

            if weights:
                base_model = resnet18(weights=weights)

            feature_dim = base_model.fc.in_features
        else:
            raise NotImplementedError(f"模型 {name} 未實現")

        # 移除原始分類頭
        base_model.fc = nn.Identity()
        self.encoder = base_model

        # 添加投影頭 (用於 SupCon Loss)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim)
        )
        logger.info(
            f"模型 {name} 加載完成。 特徵維度: {feature_dim}, 投影頭輸出維度: {embedding_dim}")

    def forward(self, x):
        """前向傳播，返回歸一化的嵌入向量 (用於計算 Loss)"""
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        # 返回 L2 歸一化的嵌入，這對比學習很重要
        return F.normalize(embeddings, dim=1)

    def get_features(self, x):
        """用於獲取最終用於聚類的特徵 (投影頭之前)"""
        # 在推理/聚類階段調用此方法
        with torch.no_grad():  # 確保在推理時不計算梯度
            features = self.encoder(x)
        return features
