"""
(推測功能) 將 clustered_tiles/ 目錄下已分類的圖片還原 (複製) 回 capture/ 目錄。

功能:
- 讀取 clustered_tiles/ 下所有子目錄中的圖片。
- 將這些圖片複製回 capture/ 目錄，可能用於重新聚類或其他目的。
- 可能會先清空 capture/ 目錄。

用法 (推測):
直接運行此腳本:
  python tools/restore_capture_from_clusters.py

執行方式:
  python tools/restore_capture_from_clusters.py

功能:
  - 遍歷 clustered_tiles/ 目錄下的所有子資料夾 (如 cluster_0, noise 等)。
  - 將每個子資料夾中的所有檔案複製回 capture/ 目錄。
  - 不會刪除 clustered_tiles 或其子目錄。
  - 不會檢查 capture/ 中是否已存在同名檔案 (會覆蓋)。
"""

import os
import shutil
import logging

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定目錄 ---
CLUSTERED_DIR = "clustered_tiles"  # 包含聚類結果的目錄
CAPTURE_DIR = "capture"        # 原始擷取圖片的目錄 (目標目錄)
# --- ---


def main():
    logger.info(f"--- 開始將圖片從 {CLUSTERED_DIR} 複製回 {CAPTURE_DIR} ---")

    if not os.path.isdir(CLUSTERED_DIR):
        logger.error(f"錯誤：找不到來源目錄 '{CLUSTERED_DIR}'")
        return

    if not os.path.isdir(CAPTURE_DIR):
        logger.error(f"錯誤：找不到目標目錄 '{CAPTURE_DIR}'，請確保該目錄存在。")
        # 或者可以選擇創建它: os.makedirs(CAPTURE_DIR)
        return

    copied_count = 0
    error_count = 0

    # 遍歷 clustered_tiles 下的項目 (子資料夾)
    for cluster_folder_name in os.listdir(CLUSTERED_DIR):
        cluster_folder_path = os.path.join(CLUSTERED_DIR, cluster_folder_name)

        # 確保是資料夾
        if os.path.isdir(cluster_folder_path):
            logger.info(f"處理資料夾: {cluster_folder_name}")
            # 遍歷資料夾內的檔案
            for filename in os.listdir(cluster_folder_path):
                src_path = os.path.join(cluster_folder_path, filename)
                dst_path = os.path.join(CAPTURE_DIR, filename)

                # 確保是檔案
                if os.path.isfile(src_path):
                    try:
                        shutil.copy2(src_path, dst_path)  # 使用 copy2 保留元數據
                        copied_count += 1
                        logger.debug(f"  Copied: {filename} -> {CAPTURE_DIR}/")
                    except Exception as e:
                        logger.error(
                            f"無法複製檔案 {src_path} 到 {dst_path}: {e}", exc_info=True)
                        error_count += 1
                else:
                    logger.debug(f"  跳過非檔案項目: {src_path}")
        else:
            logger.debug(f"跳過非資料夾項目: {cluster_folder_path}")

    logger.info(f"--- 複製完成 --- ")
    logger.info(f"總共成功複製了 {copied_count} 個檔案。")
    if error_count > 0:
        logger.error(f"複製過程中發生了 {error_count} 個錯誤。")


if __name__ == "__main__":
    main()
