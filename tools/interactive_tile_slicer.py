# tools/interactive_tile_slicer.py
"""
提供一個互動式工具，讓使用者透過滑鼠點擊來定義手牌 ROI 內每張牌的邊界框。

功能:
- 讀取 `configs/roi_config.json` 獲取預先定義的手牌區域 (ROI) 座標。
- 從 `screenshot_printwindow.png` 圖像中提取手牌 ROI。
- 顯示手牌 ROI 圖像，並引導使用者依次點擊每張牌的邊界。
  (注意：目前實現是讓使用者拖曳框選每張牌)。
- 使用滑鼠回呼函數 (`mouse_callback`) 處理點擊和拖曳事件。
- 記錄使用者為每張牌 (預期 `EXPECTED_TILES` 張) 框選的矩形座標 (x, y, w, h)。
- 將所有確認的牌框座標列表保存到 `configs/tile_slicer_config.json` 文件中。

用法:
直接運行此腳本以啟動互動式切割設定工具。
```bash
python tools/interactive_tile_slicer.py
```
運行前置條件:
- 必須已成功執行 `tools/find_roi_interactively.py` 來生成 `configs/roi_config.json`。
- 專案根目錄下必須存在 `screenshot_printwindow.png` 文件。

此工具生成的 `configs/tile_slicer_config.json` 文件包含了切割單張牌所需的精確座標，
會被 `image_processing/tile_slicer.py` 模組讀取並用於實際的圖像切割。
"""
import cv2
import os
import json
import numpy as np

# --- 設定 ---
# 設定檔路徑 (相對於此腳本)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ROI_CONFIG_FILE = os.path.join(
    _script_dir, "..", "configs", "roi_config.json")
_TILE_CONFIG_FILE = os.path.join(
    _script_dir, "..", "configs", "tile_slicer_config.json")
_SCREENSHOT_PATH = os.path.join(
    _script_dir, "..", "screenshot_printwindow.png")

EXPECTED_TILES = 14  # 標準手牌數量 (包括剛摸的牌)
WINDOW_NAME = "Interactive Tile Selector"  # Fixed window name
# --- ---

# --- 全局變數用於滑鼠回呼 ---
drawing = False         # True 如果正在拖曳滑鼠
ix, iy = -1, -1         # 起始座標
cx, cy = -1, -1         # 目前座標 (用於繪製拖曳中的矩形)
# (x, y, w, h) of the completed selection before confirmation
selection_rect = None
confirmed_boxes = []    # List to store confirmed boxes [(x,y,w,h), ...]
# Index of the tile being selected (0 to EXPECTED_TILES-1)
current_tile_index = 0
roi_display_img = None  # Image being displayed and drawn on
hand_roi_img_clean = None  # Clean copy of the hand ROI
# --- ---


def load_roi_config():
    """ 載入手牌 ROI 設定檔。 """
    if not os.path.exists(_ROI_CONFIG_FILE):
        print(f"錯誤：找不到 ROI 設定檔 '{_ROI_CONFIG_FILE}'")
        print("請先執行 tools/find_roi_interactively.py 來產生設定檔。")
        return None
    try:
        with open(_ROI_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if "hand_roi" in config and all(k in config["hand_roi"] for k in ["x", "y", "width", "height"]):
            return config["hand_roi"]
        else:
            print(f"錯誤：設定檔 '{_ROI_CONFIG_FILE}' 格式不正確。")
            return None
    except Exception as e:
        print(f"錯誤：讀取 ROI 設定檔時發生問題: {e}")
        return None


def mouse_callback(event, x, y, flags, param):
    """ 滑鼠事件回呼函式 """
    global ix, iy, cx, cy, drawing, selection_rect, roi_display_img, hand_roi_img_clean

    if event == cv2.EVENT_LBUTTONDOWN:
        # 如果還沒開始選下一個，或者允許重選當前
        if selection_rect is None:
            drawing = True
            ix, iy = x, y
            cx, cy = x, y  # Reset current position
            selection_rect = None  # Clear previous selection if starting new one

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cx, cy = x, y
            # Redraw the image with the current rectangle being dragged
            roi_display_img = hand_roi_img_clean.copy()
            draw_boxes(roi_display_img)  # Draw confirmed boxes first
            cv2.rectangle(roi_display_img, (ix, iy), (x, y),
                          (0, 255, 0), 1)  # Draw current drag

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            ex, ey = x, y
            # Ensure width and height are positive
            start_x = min(ix, ex)
            start_y = min(iy, ey)
            width = abs(ix - ex)
            height = abs(iy - ey)
            if width > 0 and height > 0:
                selection_rect = (start_x, start_y, width, height)
                print(f"框選完成 (待確認): {selection_rect}")
                # Draw the final selected rectangle (thicker, yellow)
                roi_display_img = hand_roi_img_clean.copy()
                draw_boxes(roi_display_img)
                cv2.rectangle(roi_display_img, (start_x, start_y),
                              (start_x + width, start_y + height), (0, 255, 255), 2)
            else:
                selection_rect = None  # Invalid selection (zero size)
                roi_display_img = hand_roi_img_clean.copy()  # Redraw without selection
                draw_boxes(roi_display_img)


def draw_boxes(img):
    """ 在影像上繪製已確認的框 """
    global confirmed_boxes
    for i, (x, y, w, h) in enumerate(confirmed_boxes):
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (200, 200, 200), 1)  # Gray for confirmed
        cv2.putText(img, str(i+1), (x+2, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# --- 主程式 ---
print("--- 互動式手牌分割設定工具 (滑鼠模式) ---")

# 1. 載入設定和圖片
roi_config = load_roi_config()
if roi_config is None:
    exit()

img = cv2.imread(_SCREENSHOT_PATH)
if img is None:
    print(f"錯誤：無法載入截圖 '{_SCREENSHOT_PATH}'")
    exit()

# 2. 提取手牌 ROI 區域
roi_x = roi_config["x"]
roi_y = roi_config["y"]
roi_w = roi_config["width"]
roi_h = roi_config["height"]
hand_roi_img_clean = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w].copy()

if hand_roi_img_clean.size == 0:
    print("錯誤：提取的手牌 ROI 區域為空，請檢查 ROI 座標。")
    exit()

roi_display_img = hand_roi_img_clean.copy()

# 3. 建立視窗並設定滑鼠回呼
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print(f"\n請在 '{WINDOW_NAME}' 視窗中操作：")
print("1. 用滑鼠左鍵拖曳框選第 1 張牌。")
print("2. 完成框選後，按 Enter 或 Space 確認。")
print("3. 按 c 清除當前框選 (可重畫)。")
print("4. 按 Esc 退出。")

while current_tile_index < EXPECTED_TILES:
    # 在進入 waitKey 前確保顯示最新的畫面狀態
    display_copy = roi_display_img.copy()
    # 添加提示文字
    prompt = f"Select Tile [{current_tile_index + 1}/{EXPECTED_TILES}] (Enter/Space=Confirm, c=Clear, Esc=Exit)"
    cv2.putText(display_copy, prompt, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # 如果有待確認的框，用黃色粗框提示
    if selection_rect and not drawing:
        (sx, sy, sw, sh) = selection_rect
        cv2.rectangle(display_copy, (sx, sy),
                      (sx + sw, sy + sh), (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, display_copy)
    key = cv2.waitKey(20) & 0xFF  # Use waitKey(20) for responsiveness

    if key == 27:  # Esc 鍵
        print("\n使用者要求退出。")
        break
    elif key == ord('c'):  # c 鍵 - 清除當前選擇
        print("清除當前框選。")
        selection_rect = None
        drawing = False  # Ensure drawing stops
        roi_display_img = hand_roi_img_clean.copy()  # Reset display
        draw_boxes(roi_display_img)
    elif (key == 13 or key == 32) and selection_rect is not None:  # Enter 或 Space 確認
        print(f"確認第 {current_tile_index + 1} 張牌座標: {selection_rect}")
        confirmed_boxes.append(selection_rect)
        current_tile_index += 1
        selection_rect = None  # 清除待確認框，準備選下一張
        roi_display_img = hand_roi_img_clean.copy()  # Reset display for next selection
        draw_boxes(roi_display_img)  # Draw newly confirmed box in gray

cv2.destroyAllWindows()

# 4. 儲存結果到設定檔
if len(confirmed_boxes) == EXPECTED_TILES:
    print(f"\n已成功選取所有 {EXPECTED_TILES} 張牌的座標。")
    output_data = {"tile_boxes": confirmed_boxes}
    try:
        # 確保 configs 資料夾存在
        os.makedirs(os.path.dirname(_TILE_CONFIG_FILE), exist_ok=True)
        with open(_TILE_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"手牌分割座標已成功儲存至: {_TILE_CONFIG_FILE}")
    except Exception as e:
        print(f"\n錯誤：儲存手牌分割設定檔時發生問題: {e}")
elif confirmed_boxes:  # 如果中途退出但有已確認的框
    print(f"\n儲存了已確認的 {len(confirmed_boxes)} 張牌座標。")
    output_data = {"tile_boxes": confirmed_boxes}
    try:
        os.makedirs(os.path.dirname(_TILE_CONFIG_FILE), exist_ok=True)
        with open(_TILE_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"部分手牌分割座標已成功儲存至: {_TILE_CONFIG_FILE}")
    except Exception as e:
        print(f"\n錯誤：儲存手牌分割設定檔時發生問題: {e}")
else:
    print("\n未確認任何有效的牌座標，未儲存設定檔。")

print("互動式手牌分割工具已關閉。")
