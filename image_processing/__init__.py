"""
圖像處理套件。

此套件包含用於麻將牌圖像處理的各種工具和功能，例如：
- `hand_detector`: 偵測圖像中的手牌區域。
- `tile_slicer`: 從手牌區域中切割出單張牌。
- `tile_defs`: 可能包含牌的尺寸或其他定義。

用法:
其他模組（如 `main.py` 或 `tools` 中的腳本）可以導入此套件中的特定功能。
```python
from image_processing.hand_detector import get_hand_roi
from image_processing.tile_slicer import slice_hand_roi
```
"""
# 可選地導入子模組以方便外部調用
# from .tile_slicer import slice_hand_roi
