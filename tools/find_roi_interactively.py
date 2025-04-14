# find_roi_interactively.py
"""
提供一個互動式工具，讓使用者可以在螢幕截圖上選擇感興趣區域 (ROI)。

功能:
- 顯示螢幕截圖。
- 允許使用者透過滑鼠拖拽或其他方式選定一個矩形區域。
- 將選定的區域座標保存到設定檔 (例如 configs/roi_config.json)。

用法:
直接運行此腳本以啟動互動式 ROI 選擇工具:
  python tools/find_roi_interactively.py

執行方式:
  python tools/find_roi_interactively.py

其他模組如何使用:
  - image_processing/hand_detector.py 模組會讀取此工具產生的
    configs/roi_config.json 檔案來確定從遊戲畫面中提取哪一部分作為手牌區域。
"""
import cv2
import os
import json  # Import json module

# --- 設定 ---
# 要分析的圖片路徑 (相對於此腳本)
script_dir = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(script_dir, "..", "screenshot_printwindow.png")
# 設定檔輸出路徑 (相對於此腳本)
CONFIG_DIR = os.path.join(script_dir, "..", "configs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "roi_config.json")
# --- ---

# 檢查圖片是否存在
if not os.path.exists(IMAGE_PATH):
    # Provide more context in error message
    abs_path = os.path.abspath(IMAGE_PATH)
    print(f"錯誤：找不到圖片檔案 '{IMAGE_PATH}' (預期路徑: {abs_path})")
    print("請確保 screenshot_printwindow.png 在專案根目錄下。")
    exit()

# 載入圖片
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"錯誤：無法使用 OpenCV 載入圖片 '{IMAGE_PATH}'")
    exit()

print("\n--- 請在彈出的視窗中框選手牌區域 ---")
print("1. 使用滑鼠左鍵拖曳一個矩形框，包住所有的手牌。")
print("2. 框選完成後，按下 Enter 鍵 或 空白鍵。")
print("3. 如果要取消框選並重新開始，按下 c 鍵。")
print("4. 如果要直接退出，按下 Esc 鍵。")
print("(視窗可能會有點大，您可以先調整視窗大小)")

# 使用 selectROI 讓使用者互動式選取
# 'selectROI' 會開啟一個視窗
# False 表示不顯示十字線
# Use ASCII title for better compatibility
window_title = "Select Hand ROI (Press Enter/Space to confirm, Esc to cancel)"
r = cv2.selectROI(window_title, img,
                  fromCenter=False, showCrosshair=True)

# selectROI 返回 (x, y, w, h) tuple
# 如果使用者按下 Esc，則返回 (0, 0, 0, 0)
if r == (0, 0, 0, 0):
    print("\n使用者取消了操作或直接退出。")
else:
    roi_x, roi_y, roi_w, roi_h = r
    print("\n--- 互動式選取結果 ---")
    # Format the data for JSON
    roi_data = {
        "hand_roi": {
            "x": roi_x,
            "y": roi_y,
            "width": roi_w,
            "height": roi_h
        }
    }
    # Ensure the configs directory exists
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        # Write data to JSON file
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(roi_data, f, indent=4)
        print(f"ROI 座標已成功儲存至: {CONFIG_FILE}")
        print(f"HAND_ROI_X = {roi_x}")
        print(f"HAND_ROI_Y = {roi_y}")
        print(f"HAND_ROI_WIDTH = {roi_w}")
        print(f"HAND_ROI_HEIGHT = {roi_h}")

    except Exception as e:
        print(f"\n錯誤：儲存 ROI 設定檔時發生問題: {e}")

    # (可選) 顯示選取的區域以供預覽
    # cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    # cv2.imshow("Selected ROI Preview", cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))
    # cv2.waitKey(0)

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
print("互動式選取工具已關閉。")
