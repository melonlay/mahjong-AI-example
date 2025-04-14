"""
定義應用程式 GUI 中的圖像顯示區域。

此模組包含 `ImageDisplayFrame` 類，該類是一個 `ttk.Frame` 子類，
專門用於在 Tkinter 介面中展示圖像。
它接收 OpenCV 格式的圖像 (NumPy BGR array)，將其轉換並顯示在一個 Label 元件中。
提供 `update_image` 方法來動態更新顯示的內容。

用法:
主要由 `gui.main_window.MainWindow` 實例化並嵌入其佈局中。
```python
from gui.image_display import ImageDisplayFrame

# 在 MainWindow 的 __init__ 中:
# self.image_display = ImageDisplayFrame(self.right_frame, initial_width=640, initial_height=360)
# self.image_display.pack(fill=tk.BOTH, expand=True)

# 更新圖像 (例如從擷取線程):
# new_cv_image = get_latest_frame()
# self.image_display.update_image(new_cv_image)
```
也可以直接運行此文件進行簡單的單元測試 (需要一個名為 `screenshot_printwindow.png` 的圖片在上一層目錄):
```bash
python gui/image_display.py
```
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageDisplayFrame(ttk.Frame):
    """
    一個顯示 OpenCV 影像 (NumPy array in BGR format) 的 Tkinter Frame。
    提供 update_image 方法來更新顯示的影像。
    """

    def __init__(self, parent, initial_width=320, initial_height=180, *args, **kwargs):
        """
        初始化 Frame。

        Args:
            parent: 父層 Tkinter 元件。
            initial_width: 初始顯示寬度。
            initial_height: 初始顯示高度。
        """
        super().__init__(parent, *args, **kwargs)
        self.display_width = initial_width
        self.display_height = initial_height

        # 使用 Label 來顯示圖片
        self.image_label = ttk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 預設顯示一個空白圖片
        self.placeholder_image = np.zeros(
            (initial_height, initial_width, 3), dtype=np.uint8)
        self.update_image(self.placeholder_image)

    def update_image(self, cv_image: np.ndarray):
        """
        更新顯示的 OpenCV 影像。

        Args:
            cv_image: OpenCV 影像 (NumPy array, BGR 格式)。
        """
        if cv_image is None:
            # 如果傳入 None，顯示預設圖
            img_to_show = self.placeholder_image
        else:
            img_to_show = cv_image

        # 確保影像非空
        if img_to_show.shape[0] == 0 or img_to_show.shape[1] == 0:
            img_to_show = self.placeholder_image  # Use placeholder if invalid shape

        # 縮放圖片以符合顯示大小
        # 使用 INTER_AREA 進行縮小以獲得較好品質
        resized_image = cv2.resize(
            img_to_show, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)

        # OpenCV (BGR) -> Pillow (RGB)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Pillow -> Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # 更新 Label 的圖片
        self.image_label.config(image=tk_image)
        # *** 重要: 必須保留對 tk_image 的引用，否則會被垃圾回收 ***
        self.image_label.image = tk_image


# --- 測試用 ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Display Test")
    root.geometry("400x300")

    # 建立顯示框架
    display_frame = ImageDisplayFrame(
        root, initial_width=380, initial_height=280)
    display_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # 載入一個範例圖片 (例如上次的截圖)
    try:
        # Assuming the test image is in the parent directory
        test_img = cv2.imread("../screenshot_printwindow.png")
        if test_img is not None:
            print("載入測試圖片成功")
            # 等待一下再更新，確保視窗先顯示
            root.after(500, lambda: display_frame.update_image(test_img))
        else:
            print("錯誤：無法載入測試圖片 '../screenshot_printwindow.png'。請確保檔案存在。")
            # 即使載入失敗，也會顯示預設的黑色圖片
    except Exception as e:
        print(f"載入測試圖片時發生錯誤: {e}")

    root.mainloop()
