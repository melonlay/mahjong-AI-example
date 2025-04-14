"""
定義麻將牌相關的常量和數據結構。

此模組用於集中管理項目中關於麻將牌的固定定義，
例如牌的種類、名稱、可能的尺寸常量等。
目的是提供一個統一的、一致的參考點，供其他模組（如訓練、分類、GUI）使用。

目前此文件是個佔位符，未來可以根據需要添加實際的定義。

用法:
當需要引用麻將牌的標準定義時，從此模組導入。
```python
# 假設未來定義了這些常量:
# from image_processing.tile_defs import ALL_TILE_NAMES, STANDARD_TILE_HEIGHT

# print("所有牌的名稱:", ALL_TILE_NAMES)
```
此文件本身不能直接運行以產生功能。
"""

# 暫時留空，後續會加入牌的定義
# 例如：
# TILE_TYPES = {
#     "1m": "一萬",
#     "2m": "二萬",
#     ...
#     "1p": "一筒",
#     ...
#     "1s": "一索",
#     ...
#     "ew": "東風", # East Wind
#     ...
#     "rd": "紅中", # Red Dragon
#     ...
# }

# 可以定義標準的內部表示法
# e.g., using strings like "1m", "5pr" (red 5 pin), "ew"

if __name__ == '__main__':
    print("此檔案定義麻將牌種類，目前為空。")
