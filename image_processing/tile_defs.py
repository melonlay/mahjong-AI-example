"""
定義與麻將牌相關的常量、枚舉或數據結構。

功能:
- 可能包含牌的種類列表、數值映射、尺寸常量等。
- 提供項目中其他模組一致的牌定義參考。

用法:
在需要引用牌定義的地方導入。
  from image_processing.tile_defs import TILE_TYPES, TILE_WIDTH # 假設的常量

  print(f"總共有 {len(TILE_TYPES)} 種牌")
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
