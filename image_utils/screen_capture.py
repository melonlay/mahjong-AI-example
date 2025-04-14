"""
提供擷取特定視窗畫面內容的功能，特別針對遊戲視窗。

主要使用 Windows API 的 PrintWindow 功能，以提高對使用非 GDI 渲染
(如 DirectX, OpenGL) 的視窗的擷取準確性和可靠性。

主要函式:
    capture_game_window: 尋找指定標題的視窗並回傳其畫面內容 (OpenCV 格式)。
"""
import pygetwindow as gw
import time
import numpy as np
import cv2
import ctypes
import ctypes.wintypes as wintypes  # 引入 Windows 型別
import platform  # To check OS
import logging  # 新增

# 建立此模組的 logger
logger = logging.getLogger(__name__)

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    # --- Windows API 相關設定 ---
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # Window Show State Constants
    SW_RESTORE = 9

    # PrintWindow Flags
    PW_CLIENTONLY = 0x1  # 只擷取客戶區 (不含標題列和邊框) - 通常建議使用
    PW_RENDERFULLCONTENT = 0x2  # (Windows 8.1+) 嘗試渲染所有內容

    # WinAPI 函數原型定義 (提高可讀性和安全性)
    # --- Let ctypes and pygetwindow handle GetClientRect/GetWindowRect argument types ---
    # user32.GetClientRect.argtypes = [
    #     wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    # user32.GetClientRect.restype = wintypes.BOOL
    # user32.GetWindowRect.argtypes = [
    #     wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    # user32.GetWindowRect.restype = wintypes.BOOL
    # --- Keep definitions for functions primarily used by capture_game_window ---
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.GetObjectW.argtypes = [wintypes.HGDIOBJ,
                                 ctypes.c_int, wintypes.LPVOID]  # Generic pointer
    gdi32.GetObjectW.restype = ctypes.c_int
    gdi32.GetBitmapBits.argtypes = [
        wintypes.HBITMAP, wintypes.LONG, wintypes.LPVOID]
    gdi32.GetBitmapBits.restype = wintypes.LONG
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL

    # Define BITMAP structure
    class BITMAP(ctypes.Structure):
        _fields_ = [
            ('bmType', wintypes.LONG),
            ('bmWidth', wintypes.LONG),
            ('bmHeight', wintypes.LONG),
            ('bmWidthBytes', wintypes.LONG),
            ('bmPlanes', wintypes.WORD),
            ('bmBitsPixel', wintypes.WORD),
            ('bmBits', wintypes.LPVOID),
        ]
    # Correct GetObjectW signature using the defined BITMAP structure
    gdi32.GetObjectW.argtypes = [wintypes.HBITMAP,
                                 ctypes.c_int, ctypes.POINTER(BITMAP)]
else:
    logger.warning("非 Windows 系統，PrintWindow 功能不可用。")  # print 改為 warning

# --- 修改後的函式 ---


def capture_game_window(window_title: str, use_client_area=True) -> np.ndarray | None:
    """
    尋找指定標題的視窗並使用 PrintWindow API (僅限 Windows) 擷取其內容。

    Args:
        window_title: 目標視窗的精確標題。
        use_client_area: 是否只擷取客戶區 (不含標題列和邊框)。預設為 True。

    Returns:
        如果成功，回傳 OpenCV 格式 (BGR) 的截圖影像 (NumPy array)。
        如果找不到視窗、非 Windows 系統或發生錯誤，則回傳 None。
    """
    if not IS_WINDOWS:
        logger.error("此功能僅支援 Windows 系統。")  # print 改為 error
        return None

    hwnd = None
    hwnd_dc = None
    mem_dc = None
    bitmap = None
    old_bitmap = None  # To store the original bitmap from mem_dc

    try:
        # 1. 尋找視窗句柄 (HWND) - 使用更嚴格的匹配
        # print 改為 debug
        logger.debug(f"正在尋找標題 *完全等於* '{window_title}' 的視窗...")
        all_windows = gw.getAllWindows()
        target_window_obj = None
        logger.debug("--- 偵測到的視窗標題 ---")  # print 改為 debug
        found_titles = []
        for w in all_windows:
            # logger.debug(f"  - '{w.title}'") # Detailed debug
            found_titles.append(w.title)
            if w.title == window_title:
                target_window_obj = w
                logger.debug(f"  >>> 精確匹配找到: '{w.title}'")  # print 改為 debug
                break  # 找到第一個完全符合的就停止
        # Print all titles at once after loop (at debug level)
        for title in found_titles:
            logger.debug(f"  - '{title}'")  # print 改為 debug
        logger.debug("-------------------------")  # print 改為 debug

        if target_window_obj is None:
            # print 改為 error
            logger.error(f"找不到標題 *完全等於* '{window_title}' 的視窗。")
            logger.error("請檢查：")
            logger.error("  1. 遊戲是否正在執行？")
            logger.error(
                f"  2. main.py 中的 WINDOW_TITLE 是否與遊戲視窗標題 *完全* 一致 (包含空格)？")
            logger.error(
                "  3. 上方列出的偵測到的標題是否包含您要找的視窗？ (需將 logging level 設為 DEBUG 查看)")
            return None

        # Ensure we have a valid window handle from pygetwindow
        if not hasattr(target_window_obj, '_hWnd'):
            logger.error(f"無法從 pygetwindow 物件獲取視窗句柄。")  # print 改為 error
            return None
        hwnd = target_window_obj._hWnd

        if not user32.IsWindow(hwnd):
            logger.error(f"找到的視窗句柄 {hwnd} 無效或視窗已關閉。")  # print 改為 error
            return None

        # --- 檢查視窗狀態 ---
        if user32.IsIconic(hwnd):
            logger.info("視窗已最小化，正在嘗試恢復...")  # print 改為 info
            user32.ShowWindow(hwnd, SW_RESTORE)  # Use constant
            time.sleep(0.5)  # Give window time to restore
            if user32.IsIconic(hwnd):  # Check again
                logger.warning("恢復最小化視窗失敗。擷取可能會失敗或為空。")  # print 改為 warning

        # 2. 獲取視窗尺寸 (using ctypes directly now)
        rect = wintypes.RECT()
        if use_client_area:
            # Call GetClientRect without pre-set argtypes
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                error_code = ctypes.GetLastError()
                # print 改為 error
                logger.error(f"無法獲取視窗客戶區尺寸 (GetClientRect)。錯誤碼: {error_code}")
                return None
            width = rect.right - rect.left
            height = rect.bottom - rect.top
        else:
            # Call GetWindowRect without pre-set argtypes
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                error_code = ctypes.GetLastError()
                # print 改為 error
                logger.error(f"無法獲取視窗尺寸 (GetWindowRect)。錯誤碼: {error_code}")
                return None
            width = rect.right - rect.left
            height = rect.bottom - rect.top

        if width <= 0 or height <= 0:
            logger.error(  # print 改為 error
                f"獲取的視窗尺寸不正確 (Width: {width}, Height: {height})。可能是視窗尚未完全渲染或被隱藏。")
            return None
        # 新增 debug
        logger.debug(
            f"獲取視窗尺寸 (use_client={use_client_area}): W={width}, H={height}")

        # 3. 創建 GDI 物件
        hwnd_dc = user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            # print 改為 error
            logger.error(
                f"無法獲取視窗 DC (GetWindowDC)。錯誤碼: {ctypes.GetLastError()}")
            return None

        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        if not mem_dc:
            logger.error(  # print 改為 error
                f"無法創建內存 DC (CreateCompatibleDC)。錯誤碼: {ctypes.GetLastError()}")
            user32.ReleaseDC(hwnd, hwnd_dc)
            hwnd_dc = None
            return None

        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        if not bitmap:
            logger.error(  # print 改為 error
                f"無法創建兼容位圖 (CreateCompatibleBitmap)。錯誤碼: {ctypes.GetLastError()}")
            gdi32.DeleteDC(mem_dc)
            mem_dc = None
            user32.ReleaseDC(hwnd, hwnd_dc)
            hwnd_dc = None
            return None

        old_bitmap = gdi32.SelectObject(mem_dc, bitmap)
        if not old_bitmap:
            logger.error(  # print 改為 error
                f"無法將位圖選入內存 DC (SelectObject)。錯誤碼: {ctypes.GetLastError()}")
            gdi32.DeleteObject(bitmap)
            bitmap = None
            gdi32.DeleteDC(mem_dc)
            mem_dc = None
            user32.ReleaseDC(hwnd, hwnd_dc)
            hwnd_dc = None
            return None
        logger.debug("GDI 物件創建並選取成功。")  # 新增 debug

        # 4. 執行 PrintWindow
        # flags = PW_CLIENTONLY if use_client_area else 0 # Original more compatible flags
        # Combine flags for potentially better results with non-GDI apps
        flags = 0
        if use_client_area:
            flags |= PW_CLIENTONLY
        # Always try to render full content on Win 8.1+
        flags |= PW_RENDERFULLCONTENT

        # print 改為 debug
        logger.debug(
            f"呼叫 PrintWindow (hwnd={hwnd}, hdc={mem_dc}, flags={flags})...")
        success = user32.PrintWindow(hwnd, mem_dc, flags)
        logger.debug(f"PrintWindow 返回: {success}")  # print 改為 debug

        if not success:
            error_code = ctypes.GetLastError()
            logger.error(f"PrintWindow 失敗。錯誤碼: {error_code}")  # print 改為 error
            # Fall through to cleanup in finally block
            return None

        # 5. 從位圖獲取數據
        bitmap_info = BITMAP()
        if gdi32.GetObjectW(bitmap, ctypes.sizeof(bitmap_info), ctypes.byref(bitmap_info)) == 0:
            # print 改為 error
            logger.error(f"GetObjectW 無法獲取位圖資訊。 錯誤碼: {ctypes.GetLastError()}")
            return None

        bits_per_pixel = bitmap_info.bmBitsPixel
        buffer_size = height * bitmap_info.bmWidthBytes
        if buffer_size <= 0:
            logger.error(f"計算出的緩衝區大小無效 ({buffer_size})。")  # print 改為 error
            return None

        bitmap_buffer = (ctypes.c_ubyte * buffer_size)()
        if gdi32.GetBitmapBits(bitmap, buffer_size, bitmap_buffer) == 0:
            # print 改為 error
            logger.error(
                f"GetBitmapBits 無法獲取位圖數據。 錯誤碼: {ctypes.GetLastError()}")
            return None
        logger.debug("位圖數據獲取成功。")  # 新增 debug

        # 6. 將數據轉換為 OpenCV 格式 (BGR)
        # Check the pixel format to determine the correct OpenCV conversion
        if bits_per_pixel == 32:
            # Assume BGRA format from PrintWindow
            image = np.frombuffer(bitmap_buffer, dtype=np.uint8).reshape(
                (height, width, 4))
            # Convert BGRA to BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            logger.debug("從 32bpp (BGRA) 緩衝區轉換為 BGR 影像。")  # 新增 debug
        elif bits_per_pixel == 24:
            # Assume BGR format
            image = np.frombuffer(bitmap_buffer, dtype=np.uint8).reshape(
                (height, width, 3))
            img_bgr = image  # Already BGR
            logger.debug("從 24bpp (BGR) 緩衝區創建 BGR 影像。")  # 新增 debug
        else:
            logger.error(f"不支援的位元深度 (bits per pixel): {bits_per_pixel}")
            return None

        return img_bgr

    except Exception as e:
        logger.error(f"擷取視窗畫面時發生未預期錯誤: {e}", exc_info=True)
        return None

    finally:
        # 7. 清理 GDI 物件 (非常重要)
        if mem_dc and old_bitmap:
            gdi32.SelectObject(mem_dc, old_bitmap)
            logger.debug("已恢復內存 DC 的原始位圖。")  # 新增 debug
        if bitmap:
            gdi32.DeleteObject(bitmap)
            logger.debug("已刪除兼容位圖。")  # 新增 debug
            bitmap = None
        if mem_dc:
            gdi32.DeleteDC(mem_dc)
            logger.debug("已刪除內存 DC。")  # 新增 debug
            mem_dc = None
        if hwnd_dc:
            user32.ReleaseDC(hwnd, hwnd_dc)
            logger.debug("已釋放視窗 DC。")  # 新增 debug
            hwnd_dc = None
        logger.debug("GDI 資源清理完成。")  # 新增 debug


# --- 測試用 (僅 Windows) ---
if __name__ == "__main__" and IS_WINDOWS:
    logging.basicConfig(level=logging.DEBUG)  # 在測試時設定為 DEBUG
    logger.info("--- 開始 screen_capture.py 測試 ---")

    # test_title = "小算盤"  # 測試小算盤
    test_title = "麻雀一番街"  # 測試遊戲
    # test_title = "記事本"    # 測試記事本
    # test_title = "Microsoft Edge" # 測試 Edge

    logger.info(f"測試擷取視窗: '{test_title}'")
    start_time = time.time()
    captured_image = capture_game_window(test_title, use_client_area=True)
    # captured_image = capture_game_window(test_title, use_client_area=False) # 測試擷取完整視窗
    end_time = time.time()

    if captured_image is not None:
        logger.info(f"擷取成功！ 耗時: {end_time - start_time:.4f} 秒")
        try:
            cv2.imshow(f"Captured Image: {test_title}", captured_image)
            logger.info("顯示擷取到的圖片，按任意鍵關閉...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Save the image for inspection
            save_path = "captured_test_image.png"
            cv2.imwrite(save_path, captured_image)
            logger.info(f"測試圖片已儲存至: {save_path}")

        except Exception as e:
            logger.error(f"顯示或儲存圖片時出錯: {e}", exc_info=True)
    else:
        logger.error("擷取失敗。")

    logger.info("--- 結束 screen_capture.py 測試 ---")
