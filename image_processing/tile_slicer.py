"""
麻將牌圖像切割模組。

功能:
- 從輸入的手牌區域 (ROI) 圖像中，根據設定檔 (`configs/tile_slicer_config.json`) 
  中預定義的邊界框 (bounding boxes) 切割出單張麻將牌。
- 主要函數 `slice_hand_roi` 負責載入設定並執行切割操作。
- 內部使用 `_load_tile_config` 函數來快取和載入 JSON 設定檔，並進行基本驗證。
- 對於超出 ROI 範圍的邊界框會進行裁剪處理。

用法:
在獲得手牌 ROI 圖像後調用，通常用於準備訓練數據或進行單牌分析。
```python
import cv2
from image_processing.hand_detector import get_hand_roi
from image_processing.tile_slicer import slice_hand_roi

screenshot = cv2.imread('full_game_screenshot.png')
if screenshot is not None:
    hand_roi = get_hand_roi(screenshot)
    if hand_roi is not None:
        tile_images = slice_hand_roi(hand_roi) # 從設定檔讀取切割座標
        if tile_images:
            print(f"成功切割出 {len(tile_images)} 張牌圖像。")
            # 可以將 tile_images 用於後續處理，例如保存或傳遞給模型
            # for i, tile in enumerate(tile_images):
            #     cv2.imwrite(f'tile_{i+1}.png', tile)
        else:
            print("未能根據設定檔切割出手牌圖像。")
    else:
        print("無法偵測手牌區域。")
else:
    print("無法讀取截圖。")
```
也可以直接運行此文件進行簡單的單元測試 (需要相應的設定檔和上一層目錄的截圖):
```bash
python image_processing/tile_slicer.py
```
"""
import cv2
import numpy as np
from typing import List
import os
import json
import logging

# 建立此模組的 logger
logger = logging.getLogger(__name__)

# --- 設定檔路徑 (相對於此腳本) ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_TILE_CONFIG_FILE = os.path.join(
    _script_dir, "..", "configs", "tile_slicer_config.json")

# 全局變數快取設定
_cached_tile_boxes = None


def _load_tile_config():
    """ 載入 tile_slicer 設定檔，如果尚未快取。 """
    global _cached_tile_boxes
    if _cached_tile_boxes is None:
        logger.debug(f"嘗試載入 Tile 設定檔: {_TILE_CONFIG_FILE}")
        if not os.path.exists(_TILE_CONFIG_FILE):
            logger.error(f"找不到手牌分割設定檔 '{_TILE_CONFIG_FILE}'")
            logger.error("請先執行 tools/interactive_tile_slicer.py 來產生設定檔。")
            _cached_tile_boxes = []  # Return empty list on error
            return _cached_tile_boxes
        try:
            with open(_TILE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 驗證格式
            if "tile_boxes" in config and isinstance(config["tile_boxes"], list):
                # 進一步驗證列表內元素是否為長度為 4 的 list/tuple
                valid_boxes = []
                all_valid = True
                for i, box in enumerate(config["tile_boxes"]):
                    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, int) for v in box):
                        valid_boxes.append(tuple(box))  # Convert to tuple
                    else:
                        logger.warning(f"設定檔中包含無效的 box 格式 (索引 {i}): {box}")
                        all_valid = False
                _cached_tile_boxes = valid_boxes
                if all_valid:
                    logger.info(f"成功載入並驗證手牌分割設定檔: {_TILE_CONFIG_FILE}")
                else:
                    logger.warning(f"載入設定檔完成，但包含部分無效格式。")

            else:
                logger.error(
                    f"設定檔 '{_TILE_CONFIG_FILE}' 格式不正確，缺少 'tile_boxes' 列表。")
                _cached_tile_boxes = []
        except Exception as e:
            logger.error(f"讀取或解析手牌分割設定檔時發生問題: {e}", exc_info=True)
            _cached_tile_boxes = []
    return _cached_tile_boxes


def slice_hand_roi(hand_roi_image: np.ndarray) -> List[np.ndarray]:
    """
    根據 configs/tile_slicer_config.json 中的座標分割手牌區域影像。

    Args:
        hand_roi_image: 只包含手牌區域的 OpenCV 影像 (BGR)。

    Returns:
        一個包含單張麻將牌影像 (NumPy array) 的列表 (按設定檔順序)。
        如果輸入無效或找不到設定檔/座標，則回傳空列表。
    """
    if hand_roi_image is None or hand_roi_image.size == 0:
        logger.error("輸入的手牌 ROI 影像無效。")
        return []

    # 載入預定義的分割座標
    tile_boxes = _load_tile_config()
    if not tile_boxes:
        logger.error("未能從設定檔載入有效的分割座標。")
        return []

    tile_images = []
    roi_h, roi_w = hand_roi_image.shape[:2]
    logger.debug(f"手牌 ROI 尺寸: W={roi_w}, H={roi_h}")

    # 根據設定檔中的座標進行切割
    for i, (x, y, w, h) in enumerate(tile_boxes):
        # 儲存原始座標以供日誌記錄
        orig_x, orig_y, orig_w, orig_h = x, y, w, h

        # 裁剪座標以確保它們在 ROI 範圍內
        clipped = False
        if x < 0:
            w += x  # Adjust width first
            x = 0
            clipped = True
        if y < 0:
            h += y  # Adjust height first
            y = 0
            clipped = True
        if x + w > roi_w:
            w = roi_w - x  # Adjust width to fit
            clipped = True
        if y + h > roi_h:
            h = roi_h - y  # Adjust height to fit
            clipped = True

        # 如果裁剪後的寬或高變為無效，則跳過
        if w <= 0 or h <= 0:
            logger.warning(
                f"設定檔中的第 {i+1} 個座標 {(orig_x, orig_y, orig_w, orig_h)} 裁剪後無效 (w={w}, h={h}), ROI 尺寸: {(roi_w, roi_h)}")
            continue  # Skip if clipping results in invalid dimensions

        # 如果座標被裁剪，則印出提示訊息 (Debug level)
        if clipped:
            logger.debug(
                f"設定檔中的第 {i+1} 個座標 {(orig_x, orig_y, orig_w, orig_h)} 超出 ROI 範圍，已裁剪為 {(x, y, w, h)}")

        # 提取單張牌的影像
        try:  # 加入 try-except 以防裁剪後的座標仍導致 OpenCV 錯誤
            tile_img = hand_roi_image[y:y+h, x:x+w]
            if tile_img is None or tile_img.size == 0:
                logger.warning(
                    f"裁剪後的第 {i+1} 張牌影像為空，跳過。座標: {(x, y, w, h)}")
                continue
            tile_images.append(tile_img)
            logger.debug(f"成功裁剪第 {i+1} 張牌，尺寸: {tile_img.shape}")
        except Exception as e_slice:
            logger.error(
                f"裁剪第 {i+1} 張牌時發生 OpenCV 錯誤: {e_slice}，座標: {(x, y, w, h)}", exc_info=True)
            continue  # 跳過此牌

    logger.info(f"根據設定檔成功切割出 {len(tile_images)} 張牌。")
    return tile_images


# --- 測試用 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("--- 開始 tile_slicer.py 測試 ---")
    # 載入上次的截圖進行測試
    # 需要先 import hand_detector 來獲取 ROI
    try:
        from hand_detector import get_hand_roi
    except ImportError:
        logger.critical("無法匯入 hand_detector。請確保在正確的目錄下執行。", exc_info=True)
        exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(
        script_dir, "..", "screenshot_printwindow.png")
    logger.info(f"載入測試影像: {test_image_path}")
    img = cv2.imread(test_image_path)

    if img is not None:
        logger.info("影像載入成功")
        # 提取手牌 ROI (會從 roi_config.json 讀取座標)
        roi = get_hand_roi(img)

        if roi is not None:
            logger.info("手牌 ROI 提取成功")
            # 使用新的 slice_hand_roi (會從 tile_slicer_config.json 讀取座標)
            tiles = slice_hand_roi(roi)

            if tiles:
                # logger.info(f"成功分割出 {len(tiles)} 張牌。")

                # Path relative to script
                output_dir = os.path.join(script_dir, "sliced_tiles_test")
                if not os.path.exists(output_dir):
                    logger.info(f"建立測試輸出目錄: {output_dir}")
                    os.makedirs(output_dir)
                else:
                    # 清空舊的測試圖片
                    logger.info(f"清空舊的測試圖片於: {output_dir}")
                    for f in os.listdir(output_dir):
                        if f.endswith(".png"):
                            try:
                                file_path_to_remove = os.path.join(
                                    output_dir, f)
                                os.remove(file_path_to_remove)
                                logger.debug(f"已刪除舊檔案: {file_path_to_remove}")
                            except OSError as e:
                                logger.warning(f"無法刪除舊檔案 {f}: {e}")
                logger.info(f"分割出的牌將儲存至 '{output_dir}' 資料夾...")

                # --- 修改：拼接圖片而不是單獨顯示 ---
                combined_image = None
                max_h = 0
                total_w = 0
                padding = 5  # 牌之間的間隔
                valid_tiles = []

                # 先儲存並計算拼接尺寸
                for i, tile in enumerate(tiles):
                    if tile is None or tile.size == 0:
                        logger.warning(f"第 {i+1} 張牌影像為空，跳過儲存和拼接。")
                        continue
                    # 儲存單張圖片
                    try:
                        save_path = os.path.join(
                            output_dir, f"tile_{i+1}.png")
                        cv2.imwrite(save_path, tile)
                        logger.debug(f"已儲存測試牌面: {save_path}")
                    except Exception as e_write:
                        logger.warning(
                            f"儲存第 {i+1} 張牌時發生錯誤: {e_write}", exc_info=True)
                        continue  # Skip this tile if saving fails

                    # 記錄有效牌及計算尺寸
                    valid_tiles.append(tile)
                    h, w = tile.shape[:2]
                    if h > max_h:
                        max_h = h
                    total_w += w

                if not valid_tiles:
                    logger.error("沒有有效的牌面影像可以拼接。")
                else:
                    # 計算畫布總寬高
                    canvas_width = total_w + padding * (len(valid_tiles) + 1)
                    canvas_height = max_h + padding * 2
                    logger.debug(
                        f"創建拼接畫布: W={canvas_width}, H={canvas_height}")
                    # 創建白色畫布
                    combined_image = np.ones(
                        (canvas_height, canvas_width, 3), dtype=np.uint8) * 255

                    # 將牌貼到畫布上
                    current_x = padding
                    for i, tile in enumerate(valid_tiles):
                        h, w = tile.shape[:2]
                        # 計算垂直置中貼上的 Y 座標
                        start_y = padding + (max_h - h) // 2
                        try:  # Add try-except for pasting robustness
                            combined_image[start_y: start_y + h,
                                           current_x: current_x + w] = tile
                        except ValueError as e_paste:
                            logger.warning(
                                f"貼上第 {i+1} 張牌面時發生錯誤 (可能是尺寸不符): {e_paste}")
                            logger.warning(
                                f"  畫布區域: Y={start_y}:{start_y+h}, X={current_x}:{current_x+w}")
                            logger.warning(f"  牌面尺寸: H={h}, W={w}")
                        current_x += w + padding

                    # 顯示拼接後的大圖
                    try:
                        cv2.imshow("Sliced Tiles Combined", combined_image)
                        logger.info("顯示拼接後的圖片，按任意鍵關閉...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        logger.info("測試拼接圖片顯示結束。")
                    except Exception as e_show:
                        logger.error(f"顯示拼接圖片時出錯: {e_show}", exc_info=True)
                # --- 修改結束 ---

            else:
                logger.error("未能成功分割出手牌。請檢查設定檔和錯誤訊息。")
        else:
            logger.error("未能提取手牌 ROI，無法進行分割測試。")
    else:
        logger.error(f"無法載入測試影像 '{test_image_path}'。")
    logger.info("--- 結束 tile_slicer.py 測試 ---")
