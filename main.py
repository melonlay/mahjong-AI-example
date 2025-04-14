"""
主應用程式入口點。

功能:
- 初始化應用程式，例如設定 logging、加載設定。
- 創建並顯示主 GUI 視窗 (來自 gui.main_window)。
- 啟動應用程式的主事件循環。

用法:
直接運行此腳本以啟動整個應用程式:
  python main.py
"""
from image_processing.tile_slicer import slice_hand_roi
from image_processing.hand_detector import get_hand_roi
from image_utils.screen_capture import capture_game_window, IS_WINDOWS
from gui.main_window import MainWindow
import time
import threading
import tkinter as tk
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageGrab
import logging

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 從我們的模組中匯入

# --- 設定 ---
WINDOW_TITLE = "麻雀一番街"  # 要擷取的遊戲視窗標題
UPDATE_INTERVAL_MS = 100  # 更新畫面的間隔時間 (毫秒)
CAPTURE_SAVE_DIR = "capture"


class Application:
    """ 應用程式主類別，管理 GUI 和後端邏輯。 """

    def __init__(self):
        self.app_running = False
        self.capture_thread = None
        self.last_frame = None
        logger.info("應用程式初始化開始...")

        # 建立主視窗，並傳入啟動、停止和擷取手牌的回呼函式
        self.main_window = MainWindow(title="AI 控制面板 - 麻雀一番街",
                                      start_callback=self.start_capture,
                                      stop_callback=self.stop_capture,
                                      capture_hand_callback=self.capture_hand_to_files)

        # 檢查是否為 Windows，如果不是，禁用按鈕並顯示警告
        if not IS_WINDOWS:
            logger.warning("非 Windows 系統，部分功能將被禁用。")
            self.main_window.set_status_text("錯誤：非 Windows 系統")
            self.main_window.control_panel.start_button.config(
                state=tk.DISABLED)
            self.main_window.control_panel.stop_button.config(
                state=tk.DISABLED)
        logger.info("應用程式初始化完成。")

    def capture_loop(self):
        """ 在背景執行緒中不斷擷取畫面。 """
        logger.info("擷取執行緒開始...")
        while self.app_running:
            start_cap_time = time.perf_counter()
            try:
                frame = capture_game_window(WINDOW_TITLE, use_client_area=True)
                self.last_frame = frame  # 更新最新畫面
                # 可以在這裡加入圖像辨識的呼叫
                # results = process_image(frame)
                # self.update_ai_status(results)

            except Exception as e:
                logger.error(f"擷取或處理畫面時發生錯誤: {e}", exc_info=True)
                # 發生錯誤時可以考慮停止，或繼續嘗試
                # self.stop_capture() # 例如：發生錯誤時自動停止
                self.main_window.set_status_text(f"錯誤: {e}")
                time.sleep(1)  # 避免錯誤訊息刷屏
                continue  # 繼續下一輪嘗試

            end_cap_time = time.perf_counter()
            elapsed_ms = (end_cap_time - start_cap_time) * 1000
            sleep_time = max(0, (UPDATE_INTERVAL_MS - elapsed_ms) / 1000)
            logger.debug(
                f"Capture time: {elapsed_ms:.2f} ms, Sleep time: {sleep_time:.4f} s")
            if sleep_time > 0:
                time.sleep(sleep_time)
        logger.info("擷取執行緒結束。")

    def update_gui_loop(self):
        """ 定時更新 GUI 上的畫面。 """
        if self.last_frame is not None:
            self.main_window.update_display_image(self.last_frame)

        # 如果應用程式仍在運行，則安排下一次更新
        if self.app_running:
            self.main_window.after(UPDATE_INTERVAL_MS, self.update_gui_loop)

    def start_capture(self):
        """ 啟動畫面擷取和 GUI 更新。 """
        if not IS_WINDOWS:
            logger.warning("非 Windows 系統，無法啟動擷取。")
            return
        if not self.app_running:
            logger.info("啟動擷取...")
            self.app_running = True
            self.main_window.set_status_text("狀態：擷取中...")
            # 啟動背景擷取執行緒
            self.capture_thread = threading.Thread(
                target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            # 啟動 GUI 更新迴圈
            self.main_window.after(UPDATE_INTERVAL_MS, self.update_gui_loop)
        else:
            logger.info("擷取已在執行中。")

    def stop_capture(self):
        """ 停止畫面擷取。 """
        if self.app_running:
            logger.info("停止擷取...")
            self.app_running = False
            # 等待擷取執行緒結束 (可選，因為是 daemon)
            if self.capture_thread and self.capture_thread.is_alive():
                logger.info("等待擷取執行緒結束...")
                # self.capture_thread.join(timeout=1.0) # 可以設定超時
                # 通常 daemon thread 不需要 join，主程式結束時會自動退出
                pass
            self.capture_thread = None
            self.main_window.set_status_text("狀態：已停止")
            # 清除顯示畫面 (顯示預設圖)
            self.main_window.update_display_image(None)
            self.last_frame = None  # 清除最後一幀
            logger.info("擷取已停止。")
        else:
            logger.info("擷取尚未執行。")

    # --- 新增：擷取手牌並儲存的邏輯 ---
    def capture_hand_to_files(self):
        """
        擷取目前螢幕上的手牌區域，分割成單張牌並儲存到 capture 資料夾。
        """
        logger.info("執行 capture_hand_to_files...")
        if not IS_WINDOWS:
            logger.error("非 Windows 系統，無法執行擷取。")
            self.main_window.set_status_text("錯誤：僅支援 Windows")
            return

        try:
            # 1. 擷取整個螢幕 (或目標視窗，但全螢幕擷取更通用)
            # 使用 Pillow 的 ImageGrab，它通常比 capture_game_window 更適合一次性擷取
            # logger.info("擷取螢幕...")
            # screen_image_pil = ImageGrab.grab()
            # 將 PIL 影像轉換為 OpenCV 格式 (BGR)
            # screen_image_cv = cv2.cvtColor(
            #     np.array(screen_image_pil), cv2.COLOR_RGB2BGR)
            # logger.info("螢幕擷取完成")

            # --- 修改：改為直接擷取遊戲視窗客戶區 ---
            logger.info(f"擷取遊戲視窗 '{WINDOW_TITLE}' 客戶區...")
            screen_image_cv = capture_game_window(
                WINDOW_TITLE, use_client_area=True)
            if screen_image_cv is None:
                logger.error(f"無法擷取遊戲視窗 '{WINDOW_TITLE}' 的客戶區。")
                self.main_window.set_status_text("錯誤：擷取遊戲視窗失敗")
                return
            logger.info("遊戲視窗客戶區擷取完成")
            # --- 修改結束 ---

            # 2. 取得手牌 ROI
            logger.info("偵測手牌 ROI...")
            hand_roi = get_hand_roi(screen_image_cv)
            if hand_roi is None:
                logger.error("未能偵測到手牌 ROI。請確保遊戲畫面可見且 ROI 設定正確。")
                self.main_window.set_status_text("錯誤：找不到手牌區域")
                return
            logger.info("手牌 ROI 取得成功")

            # --- 新增：儲存擷取到的 ROI 以供偵錯 ---
            try:
                debug_roi_path = "debug_captured_roi.png"
                cv2.imwrite(debug_roi_path, hand_roi)
                logger.info(f"已將偵測到的手牌 ROI 儲存至: {debug_roi_path}")
            except Exception as e_roi_save:
                logger.warning(f"儲存偵錯用 ROI 影像時發生錯誤: {e_roi_save}")
            # --- 新增結束 ---

            # 3. 分割手牌
            logger.info("分割手牌...")
            tiles = slice_hand_roi(hand_roi)
            if not tiles:
                logger.error("未能成功分割出手牌。請檢查 tile_slicer_config.json 設定。")
                self.main_window.set_status_text("錯誤：分割手牌失敗")
                return
            logger.info(f"成功分割出 {len(tiles)} 張牌")

            # 4. 建立儲存目錄 (如果不存在)
            if not os.path.exists(CAPTURE_SAVE_DIR):
                logger.info(f"建立目錄: {CAPTURE_SAVE_DIR}")
                os.makedirs(CAPTURE_SAVE_DIR)

            # 5. 產生唯一檔名並儲存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 到毫秒
            logger.info(f"儲存牌面至 '{CAPTURE_SAVE_DIR}'，時間戳: {timestamp}")
            saved_count = 0
            for i, tile in enumerate(tiles):
                if tile is not None and tile.size > 0:
                    filename = os.path.join(
                        CAPTURE_SAVE_DIR, f"{timestamp}_tile_{i+1:02d}.png")
                    try:
                        cv2.imwrite(filename, tile)
                        saved_count += 1
                        logger.debug(f"已儲存: {filename}")
                    except Exception as e_save:
                        logger.error(
                            f"儲存第 {i+1} 張牌 ({filename}) 時發生錯誤: {e_save}", exc_info=True)
                else:
                    logger.warning(f"第 {i+1} 張牌影像無效，跳過儲存。")

            logger.info(f"成功儲存 {saved_count} / {len(tiles)} 張牌。")
            # 可以在狀態列顯示更詳細的成功訊息
            # self.main_window.set_status_text(f"狀態：已儲存 {saved_count} 張牌")

        except ImportError:
            logger.critical(
                "需要安裝 Pillow 才能使用螢幕擷取功能。請執行 pip install Pillow", exc_info=True)
            self.main_window.set_status_text("錯誤: Pillow 未安裝")
        except Exception as e:
            logger.error(f"擷取手牌過程中發生未預期錯誤: {e}", exc_info=True)
            # 可以在狀態列顯示錯誤
            self.main_window.set_status_text(f"錯誤: {e}")

    def run(self):
        """ 啟動 Tkinter 主迴圈。 """
        logger.info("啟動應用程式主迴圈...")
        # 確保程式結束時能停止擷取執行緒
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.main_window.mainloop()

    def on_closing(self):
        """ 關閉視窗時的處理。 """
        logger.info("視窗關閉事件觸發...")
        self.stop_capture()  # 確保執行緒停止
        self.main_window.destroy()
        logger.info("應用程式結束。")


if __name__ == "__main__":
    logger.info("==================== 應用程式啟動 ====================")
    app = Application()
    app.run()
