"""
定義用於麻將牌特徵提取的卷積神經網路 (CNN) 模型架構。

功能:
- 通常基於一個預訓練模型架構 (例如 ResNet18)。
- 修改模型以適應對比學習 (SupCon) 或其他特定任務，例如移除最後的分類層，添加投影頭 (projection head)。
- 定義模型的前向傳播邏輯。

用法:
由訓練腳本 (train.py) 導入並實例化模型。
  from trainer.clustering.model import SupConResNet # 假設模型名

  model = SupConResNet(embedding_dim=128)
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
