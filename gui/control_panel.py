"""
定義 GUI 中的控制面板組件。

功能:
- 包含應用程式的控制按鈕、設定選項、滑塊、狀態顯示等。
- 提供使用者與應用程式互動的介面。
- 將使用者操作（例如點擊按鈕）轉換為對後端功能的調用請求。

用法:
通常被 MainWindow 實例化並嵌入到主視窗佈局中。
  from gui.control_panel import ControlPanel

  # 在 MainWindow 的 __init__ 中:
  # self.control_panel = ControlPanel(self.root, self.controller)
  # self.control_panel.pack(...)
"""
import tkinter as tk
from tkinter import ttk
import logging

# 建立此模組的 logger
logger = logging.getLogger(__name__)


class ControlPanelFrame(ttk.Frame):
    """
    包含 AI 操作按鈕的 Frame。
    """

    def __init__(self, parent, start_callback=None, stop_callback=None, capture_hand_callback=None, *args, **kwargs):
        """
        初始化 Frame。

        Args:
            parent: 父層 Tkinter 元件。
            start_callback: 按下「開始」按鈕時呼叫的函式。
            stop_callback: 按下「停止」按鈕時呼叫的函式。
            capture_hand_callback: 按下「擷取手牌」按鈕時呼叫的函式。
        """
        super().__init__(parent, *args, **kwargs)
        logger.debug("初始化 ControlPanelFrame...")
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.capture_hand_callback = capture_hand_callback

        # 設定網格布局
        self.columnconfigure(0, weight=1)
        # 可以根據需要增加更多列或行的權重

        # --- 按鈕 ---
        logger.debug("建立控制按鈕...")
        self.start_button = ttk.Button(
            self, text="開始偵測", command=self.on_start_click)
        self.start_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.stop_button = ttk.Button(
            self, text="停止偵測", command=self.on_stop_click, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # --- 新增擷取手牌按鈕 ---
        self.capture_button = ttk.Button(
            self, text="擷取手牌", command=self.on_capture_hand_click)
        self.capture_button.grid(
            row=2, column=0, padx=10, pady=10, sticky="ew")

        # --- 狀態標籤 (範例) ---
        logger.debug("建立狀態標籤...")
        self.status_label = ttk.Label(self, text="狀態：待命")
        self.status_label.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # --- 其他控制項 (未來可以加入) ---
        # 例如：設定檔載入按鈕、參數調整滑桿等
        logger.debug("ControlPanelFrame 初始化完成。")

    def on_start_click(self):
        """處理「開始」按鈕點擊事件。"""
        logger.info("「開始偵測」按鈕被點擊")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.capture_button.config(state=tk.NORMAL)
        self.set_status("狀態：偵測中...")
        if self.start_callback:
            try:
                self.start_callback()
            except Exception as e:
                logger.error(f"執行 start_callback 時發生錯誤: {e}", exc_info=True)
                self.set_status(f"錯誤: {e}")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)

    def on_stop_click(self):
        """處理「停止」按鈕點擊事件。"""
        logger.info("「停止偵測」按鈕被點擊")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.capture_button.config(state=tk.NORMAL)
        self.set_status("狀態：已停止")
        if self.stop_callback:
            try:
                self.stop_callback()
            except Exception as e:
                logger.error(f"執行 stop_callback 時發生錯誤: {e}", exc_info=True)
                self.set_status(f"錯誤: {e}")

    def on_capture_hand_click(self):
        """處理「擷取手牌」按鈕點擊事件。"""
        logger.info("「擷取手牌」按鈕被點擊")
        self.capture_button.config(state=tk.DISABLED)
        self.set_status("狀態：擷取手牌中...")
        if self.capture_hand_callback:
            try:
                self.capture_hand_callback()
                self.set_status("狀態：手牌擷取完成")
            except Exception as e:
                logger.error(
                    f"呼叫 capture_hand_callback 時發生錯誤: {e}", exc_info=True)
                self.set_status(f"狀態：擷取失敗 {e}")
        else:
            logger.warning("未設定 capture_hand_callback")
            self.set_status("狀態：擷取功能未配置")
        self.capture_button.config(state=tk.NORMAL)

    def set_status(self, text: str):
        """更新狀態標籤的文字。"""
        self.status_label.config(text=text)


# --- 測試用 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("--- 開始 control_panel.py 測試 ---")
    root = tk.Tk()
    root.title("Control Panel Test")
    root.geometry("250x200")

    def dummy_start():
        logger.info("執行了 dummy_start 回呼函式")

    def dummy_stop():
        logger.info("執行了 dummy_stop 回呼函式")

    # 建立控制面板框架
    control_panel = ControlPanelFrame(
        root, start_callback=dummy_start, stop_callback=dummy_stop)
    control_panel.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    root.mainloop()
    logger.info("--- 結束 control_panel.py 測試 ---")
