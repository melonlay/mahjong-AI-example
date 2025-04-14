"""
包含用於從螢幕截圖中檢測和定位手牌區域 (ROI) 的功能。

功能:
- 可能使用圖像處理技術 (例如顏色過濾、邊緣檢測、輪廓分析) 或機器學習模型來定位手牌區域。
- 讀取 ROI 設定檔 (configs/roi_config.json)。
- 提供函數以獲取手牌區域的邊界框 (bounding box)。

用法:
可能被 GUI (例如用於初始設定 ROI) 或自動化流程調用。
  from image_processing.hand_detector import get_hand_roi

  # 假設 screenshot 是 OpenCV 讀取的圖像
  roi_rect = get_hand_roi(screenshot, config_path='configs/roi_config.json')
  if roi_rect:
      x, y, w, h = roi_rect
      hand_image = screenshot[y:y+h, x:x+w]
"""
import numpy as np
import cv2
import os
import json
import logging

# 建立此模組的 logger
logger = logging.getLogger(__name__)

# 設定檔路徑 (相對於此腳本)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_script_dir, "..", "configs", "roi_config.json")

# --- 移除手動設定座標 ---
# HAND_ROI_X = ...
# HAND_ROI_Y = ...
# HAND_ROI_WIDTH = ...
# HAND_ROI_HEIGHT = ...

# 全局變數來快取讀取的設定
_cached_roi_config = None


def _load_roi_config():
    """ 載入 ROI 設定檔，如果尚未快取。 """
    global _cached_roi_config
    if _cached_roi_config is None:
        logger.debug(f"嘗試載入 ROI 設定檔: {_CONFIG_FILE}")
        try:
            with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 驗證讀取的資料結構
            if "hand_roi" in config and all(k in config["hand_roi"] for k in ["x", "y", "width", "height"]):
                _cached_roi_config = config["hand_roi"]
                logger.info(f"成功載入 ROI 設定檔: {_CONFIG_FILE}")
            else:
                logger.error(
                    f"設定檔 '{_CONFIG_FILE}' 格式不正確，缺少 'hand_roi' 或必要鍵值。")
                _cached_roi_config = {}  # Set to empty dict to prevent reload attempts
        except FileNotFoundError:
            logger.error(f"找不到 ROI 設定檔 '{_CONFIG_FILE}'")
            logger.error("請先執行 tools/find_roi_interactively.py 來產生設定檔。")
            _cached_roi_config = {}
        except json.JSONDecodeError:
            logger.error(
                f"解析 ROI 設定檔 '{_CONFIG_FILE}' 失敗。請檢查檔案內容是否為有效的 JSON。", exc_info=True)
            _cached_roi_config = {}
        except Exception as e:
            logger.error(f"讀取 ROI 設定檔時發生未預期錯誤: {e}", exc_info=True)
            _cached_roi_config = {}
    return _cached_roi_config


def get_hand_roi(full_screenshot: np.ndarray) -> np.ndarray | None:
    """
    從完整的遊戲截圖中提取手牌區域 (Region of Interest - ROI)。
    ROI 座標從 configs/roi_config.json 檔案讀取。

    Args:
        full_screenshot: OpenCV 格式的完整遊戲畫面 (BGR)。

    Returns:
        如果成功提取，回傳手牌區域的影像 (NumPy array)。
        如果輸入影像無效、設定檔讀取失敗或 ROI 設定不正確，則回傳 None。
    """
    # 載入 ROI 設定
    roi_config = _load_roi_config()
    if not roi_config:  # 如果載入失敗或設定檔為空
        logger.error("未能載入有效的 ROI 設定，無法提取手牌區域。")
        return None

    # 從設定檔獲取座標
    roi_x = roi_config.get("x")
    roi_y = roi_config.get("y")
    roi_width = roi_config.get("width")
    roi_height = roi_config.get("height")

    # 檢查座標是否都有效
    if any(v is None or not isinstance(v, int) for v in [roi_x, roi_y, roi_width, roi_height]):
        logger.error("從設定檔讀取的 ROI 座標無效 (非整數或缺失)。")
        logger.error(f"  讀取到的設定: {roi_config}")
        return None

    # --- 後續邏輯與之前相同，使用讀取到的座標 ---
    if full_screenshot is None or full_screenshot.size == 0:
        logger.error("輸入的截圖無效 (None 或空)。")
        return None

    img_height, img_width = full_screenshot.shape[:2]
    logger.debug(f"輸入影像尺寸: W={img_width}, H={img_height}")

    # 檢查 ROI 座標是否在影像範圍內
    if (roi_x < 0 or roi_y < 0 or
        roi_x + roi_width > img_width or
        roi_y + roi_height > img_height or
            roi_width <= 0 or roi_height <= 0):
        logger.error("從設定檔讀取的 ROI 座標或尺寸超出範圍或無效。")
        logger.error(f"  影像尺寸: W={img_width}, H={img_height}")
        logger.error(
            f"  讀取設定: X={roi_x}, Y={roi_y}, W={roi_width}, H={roi_height}")
        return None

    # 使用 NumPy slicing 提取 ROI
    try:
        hand_roi = full_screenshot[roi_y: roi_y + roi_height,
                                   roi_x: roi_x + roi_width]
        logger.info(f"成功提取手牌 ROI，尺寸: {hand_roi.shape}")
        return hand_roi
    except Exception as e:
        logger.error(f"使用座標提取 ROI 時發生錯誤: {e}", exc_info=True)
        logger.error(
            f"  座標: X={roi_x}, Y={roi_y}, W={roi_width}, H={roi_height}")
        return None


# --- 測試用 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("--- 開始 hand_detector.py 測試 ---")

    # 載入上次的截圖進行測試
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(
        script_dir, "..", "screenshot_printwindow.png")
    logger.info(f"載入測試影像: {test_image_path}")
    img = cv2.imread(test_image_path)

    if img is not None:
        logger.info(f"影像載入成功，尺寸: {img.shape}")
        # 呼叫函式提取 ROI (現在會從設定檔讀取)
        roi = get_hand_roi(img)

        if roi is not None:
            # 讀取設定檔的座標用於顯示
            config = _load_roi_config()
            # 顯示原始影像和 ROI 以便核對
            try:
                cv2.imshow("Original Screenshot", cv2.resize(
                    img, (img.shape[1]//2, img.shape[0]//2)))
                cv2.imshow("Hand ROI (from config)", roi)
                logger.info("顯示預覽視窗，按任意鍵關閉...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                logger.info("預覽視窗已關閉。")
                # cv2.imwrite("hand_roi_test.png", roi)
            except Exception as e:
                logger.error(f"顯示測試影像時出錯: {e}", exc_info=True)
        else:
            logger.error("未能成功提取手牌 ROI。請檢查錯誤訊息。")
    else:
        logger.error(f"無法載入測試影像 '{test_image_path}'。")
    logger.info("--- 結束 hand_detector.py 測試 ---")
