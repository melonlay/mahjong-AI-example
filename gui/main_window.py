"""
定義應用程式的主 GUI 視窗。

此模組包含 `MainWindow` 類，該類繼承自 `tk.Tk`，構建了應用的主介面。
它負責整合各個子組件，例如圖像顯示區域 (`gui.image_display.ImageDisplayFrame`)
和控制面板 (`gui.control_panel.ControlPanelFrame`)，並將它們佈局在主視窗中。
同時，它也提供了更新子組件狀態（如顯示圖像、狀態文字）的方法。

用法:
主要由應用程式入口點 (`main.py` 或等效腳本) 實例化並啟動 Tkinter 主循環。
```python
from gui.main_window import MainWindow

app_controller = YourApplicationController() # 假設有一個控制器處理回呼

main_app = MainWindow(title="My Mahjong AI",
                      start_callback=app_controller.start,
                      stop_callback=app_controller.stop,
                      capture_hand_callback=app_controller.capture_hand)
main_app.mainloop()
```
也可以直接運行此文件進行簡單的單元測試 (需要一個名為 `screenshot_printwindow.png` 的圖片在上一層目錄):
```bash
python gui/main_window.py
```
"""
import tkinter as tk
from tkinter import ttk
from .image_display import ImageDisplayFrame
from .control_panel import ControlPanelFrame
import numpy as np  # For placeholder image type hint
import logging  # 新增

# 建立此模組的 logger
logger = logging.getLogger(__name__)


class MainWindow(tk.Tk):
    """
    應用程式主視窗，包含左側的影像顯示和右側的控制面板。
    """

    def __init__(self, title="麻雀一番街 AI", start_callback=None, stop_callback=None, capture_hand_callback=None):
        """
        初始化主視窗。

        Args:
            title: 視窗標題。
            start_callback: 傳遞給控制面板的「開始」回呼函式。
            stop_callback: 傳遞給控制面板的「停止」回呼函式。
            capture_hand_callback: 傳遞給控制面板的「擷取手牌」回呼函式。
        """
        super().__init__()
        logger.info("初始化主視窗...")  # 新增 info
        self.title(title)
        self.geometry("800x500")  # 初始大小，可以調整

        # --- 設定主視窗網格布局 ---
        self.columnconfigure(0, weight=3)  # 左側影像區域佔用較大比例
        self.columnconfigure(1, weight=1)  # 右側控制面板區域佔用較小比例
        self.rowconfigure(0, weight=1)

        # --- 建立並放置左側影像顯示框架 ---
        # 設定一個合理的初始大小給影像顯示區
        image_width = 580
        image_height = 435  # 大致維持 4:3 比例
        logger.debug("建立 ImageDisplayFrame...")  # 新增 debug
        self.image_frame = ImageDisplayFrame(
            self, initial_width=image_width, initial_height=image_height, relief=tk.SUNKEN, borderwidth=1)
        self.image_frame.grid(row=0, column=0, padx=(
            10, 5), pady=10, sticky="nsew")

        # --- 建立並放置右側控制面板框架 ---
        logger.debug("建立 ControlPanelFrame...")  # 新增 debug
        self.control_panel = ControlPanelFrame(
            self, start_callback=start_callback, stop_callback=stop_callback,
            capture_hand_callback=capture_hand_callback,
            relief=tk.RAISED, borderwidth=1)
        self.control_panel.grid(row=0, column=1, padx=(
            5, 10), pady=10, sticky="nsew")
        logger.info("主視窗初始化完成。")  # 新增 info

    def update_display_image(self, cv_image: np.ndarray | None):
        """ 更新左側顯示的影像。 """
        # logger.debug("更新顯示影像...") # 可能太頻繁，先註解
        self.image_frame.update_image(cv_image)

    def set_status_text(self, text: str):
        """ 更新右側控制面板的狀態文字。 """
        logger.debug(f"更新狀態文字: {text}")  # 新增 debug
        self.control_panel.set_status(text)


# --- 測試用 ---
if __name__ == '__main__':
    import cv2
    logging.basicConfig(level=logging.INFO)  # 測試時設為 INFO 或 DEBUG
    logger.info("--- 開始 main_window.py 測試 ---")

    def test_start():
        logger.info("Main window received start callback")  # print 改為 info
        # 模擬開始後更新狀態
        app.set_status_text("狀態：測試執行中...")
        # 模擬載入一張圖片
        try:
            # 注意相對路徑，假設從 gui 目錄執行 python -m gui.main_window
            test_img = cv2.imread("../screenshot_printwindow.png")
            if test_img is not None:
                logger.info("測試圖片載入成功，稍後更新顯示。")  # 新增 info
                # 延遲更新圖片
                app.after(200, lambda: app.update_display_image(test_img))
            else:
                logger.warning("找不到測試圖片。")  # 改為 warning
                app.set_status_text("狀態：錯誤，找不到圖片")
        except Exception as e:
            logger.error(f"載入測試圖片錯誤: {e}", exc_info=True)  # print 改為 error
            app.set_status_text("狀態：錯誤，載入圖片失敗")

    def test_stop():
        logger.info("Main window received stop callback")  # print 改為 info
        # 模擬停止後清除圖片（顯示預設圖）
        app.update_display_image(None)  # 傳入 None 會顯示預設圖
        app.set_status_text("狀態：測試已停止")

    # 建立主應用程式視窗
    app = MainWindow(start_callback=test_start, stop_callback=test_stop)
    app.mainloop()
    logger.info("--- 結束 main_window.py 測試 ---")
