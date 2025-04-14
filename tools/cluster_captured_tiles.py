"""
執行麻將牌圖像聚類的核心腳本。

功能:
- 讀取 `capture/` 目錄下的待分類圖片。
- 優先嘗試加載自訂訓練的 CNN 模型 (來自 trainer/clustering/result/) 提取特徵。
- 如果自訂模型加載失敗，則使用備選方法 (預訓練 ResNet18 + HSV 顏色直方圖) 提取特徵。
- 使用凝聚聚類 (Agglomerative Clustering) 算法對提取的特徵進行聚類。
- 將 `capture/` 目錄下的圖片根據聚類結果複製到 `clustered_tiles/` 下的對應簇子目錄中。

用法:
直接運行此腳本以執行聚類:
  python tools/cluster_captured_tiles.py

(注意: 運行前確保 `capture/` 目錄有圖片。如果希望使用自訂模型，需先完成訓練步驟)

依賴項:
  - opencv-python
  - numpy
  - scikit-learn
  - torch
  - torchvision
  - Pillow
  (請參考 PyTorch 官網安裝 torch 和 torchvision，其餘使用 pip 安裝)
"""

import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image  # torchvision transform 需要 PIL Image
import traceback

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定 ---
CAPTURE_DIR = "capture"  # 包含擷取圖片的目錄
OUTPUT_DIR = "clustered_tiles"  # 存放分類結果的目錄

# --- 通用設定 ---
RESIZE_DIM_CNN = (96, 96)  # 與訓練時的 image_size 保持一致，也用於 ResNet18 Fallback
N_CLUSTERS = 29
CLUSTER_LINKAGE = 'ward'
BATCH_SIZE = 32  # 特徵提取批次大小

# --- 自訂模型路徑 --- (指向訓練好的模型)
CUSTOM_MODEL_PATH = "trainer/clustering/result/best_mahjong_feature_extractor.pth"

# --- ResNet18 + 顏色特徵 Fallback 設定 ---
RESIZE_DIM_COLOR = (64, 64)  # Fallback 時計算顏色直方圖的大小
HSV_HIST_BINS = [8, 4, 4]  # H, S, V 的 bins 數量 (Fallback 使用)
# Fallback 特徵合併權重 (可選，這裡不使用)
# CNN_FEATURE_WEIGHT_FALLBACK = 1.0
# COLOR_FEATURE_WEIGHT_FALLBACK = 0.5

# --- 全局設定 PyTorch 設備 ---
device = None
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch 將使用設備: {device}")
except Exception as e:
    logger.critical(f"設定 PyTorch 設備時失敗: {e}", exc_info=True)
    device = None  # 標記設備設定失敗

# --- 嘗試加載自訂模型 (全局變數) ---
custom_model = None
custom_model_loaded = False
if device:
    try:
        logger.info("嘗試加載自訂訓練的特徵提取模型...")
        # 1. 創建模型結構 (ResNet18，移除分類頭)
        custom_model = models.resnet18(weights=None)
        num_ftrs = custom_model.fc.in_features
        custom_model.fc = torch.nn.Identity()

        # 2. 加載權重
        if not os.path.exists(CUSTOM_MODEL_PATH):
            raise FileNotFoundError(f"找不到自訂模型文件: {CUSTOM_MODEL_PATH}")

        state_dict = torch.load(CUSTOM_MODEL_PATH, map_location='cpu')

        # 處理可能的鍵名不匹配
        has_encoder_prefix = any(key.startswith('encoder.')
                                 for key in state_dict.keys())
        if has_encoder_prefix:
            logger.info("檢測到 'encoder.' 前綴，將進行移除...")
            new_state_dict = {k[len('encoder.'):] if k.startswith(
                'encoder.') else k: v for k, v in state_dict.items()}
            state_dict = new_state_dict

        custom_model.load_state_dict(state_dict)
        custom_model.to(device)
        custom_model.eval()
        custom_model_loaded = True
        logger.info(f"成功加載自訂模型 '{CUSTOM_MODEL_PATH}'。")

    except FileNotFoundError:
        logger.warning(
            f"未找到自訂模型文件 '{CUSTOM_MODEL_PATH}'。將使用 ResNet18 + 顏色特徵作為備選方法。")
        custom_model = None  # 確保 custom_model 為 None
    except Exception as e:
        logger.error(f"加載自訂模型時發生其他錯誤: {e}", exc_info=True)
        logger.warning("加載自訂模型失敗。將使用 ResNet18 + 顏色特徵作為備選方法。")
        custom_model = None  # 確保 custom_model 為 None
else:
    logger.error("PyTorch 設備未初始化，無法加載模型。")

# --- 圖像預處理轉換 --- #

# 自訂模型推理轉換 (與訓練時兼容)


def get_custom_inference_transforms(size=RESIZE_DIM_CNN[0]):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])

# ResNet18 Fallback 推理轉換 (使用 ImageNet 標準)


def get_resnet_inference_transforms(size=RESIZE_DIM_CNN[0]):
    # 與自訂模型使用相同的標準 ImageNet 轉換
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        # 雖然ResNet通常用224, 但這裡用96與自訂模型/訓練保持一致性可能更好
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])


custom_inference_preprocess = get_custom_inference_transforms()
resnet_inference_preprocess = get_resnet_inference_transforms()

# --- 輔助函數 --- #


def preprocess_image(image_bgr_numpy, preprocess_transform, filename="未知檔案"):
    """通用圖像預處理函數，將 BGR numpy 轉為 Tensor"""
    try:
        image_rgb_numpy = cv2.cvtColor(image_bgr_numpy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb_numpy)
        img_tensor = preprocess_transform(pil_img)
        return img_tensor
    except Exception as e:
        logger.error(f"[{filename}] 圖像預處理時出錯: {e}", exc_info=True)
        return None


def calculate_hsv_histogram(image_bgr_numpy, bins, resize_dim, filename="未知檔案"):
    """計算 HSV 顏色直方圖 (Fallback 使用)"""
    try:
        img_resized = cv2.resize(image_bgr_numpy, resize_dim)
        hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1, 2], None,
                            bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        logger.error(f"[{filename}] 計算 HSV 直方圖時出錯: {e}", exc_info=True)
        return None

# --- Fallback 聚類函數 (ResNet18 + 顏色) --- #


def cluster_with_resnet_color_features(image_paths):
    """使用預訓練 ResNet18 和 HSV 顏色直方圖進行聚類"""
    global device
    if device is None:
        logger.error("Fallback 方法需要 PyTorch 設備初始化。")
        return None

    logger.info("--- 執行備選聚類方法 (ResNet18 + HSV 顏色直方圖) ---")

    # 加載預訓練 ResNet18 模型
    try:
        logger.info("加載預訓練 ResNet18 模型...")
        resnet_model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除最後的全連接層，獲取特徵
        resnet_model.fc = torch.nn.Identity()
        resnet_model.to(device)
        resnet_model.eval()
        logger.info("預訓練 ResNet18 模型加載成功。")
    except Exception as e:
        logger.error(f"加載預訓練 ResNet18 時出錯: {e}", exc_info=True)
        return None

    cnn_features_list = []
    color_features_list = []
    processed_filepaths = []  # 記錄成功處理的文件路徑，與特徵對應
    failed_count = 0

    # 讀取圖像並提取兩種特徵
    logger.info("開始讀取圖像並提取 ResNet CNN 特徵和顏色特徵...")
    for filepath in image_paths:
        filename = os.path.basename(filepath)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            logger.warning(f"無法讀取圖片: {filepath}，跳過。")
            failed_count += 1
            continue

        # 提取顏色特徵
        color_feature = calculate_hsv_histogram(
            img_bgr, HSV_HIST_BINS, RESIZE_DIM_COLOR, filename)
        if color_feature is None:
            logger.warning(f"跳過圖片 {filename}，因顏色特徵提取失敗。")
            failed_count += 1
            continue

        # 預處理圖像以供 ResNet
        img_tensor = preprocess_image(
            img_bgr, resnet_inference_preprocess, filename)
        if img_tensor is None:
            logger.warning(f"跳過圖片 {filename}，因 CNN 預處理失敗。")
            failed_count += 1
            continue

        # 記錄成功處理的信息
        color_features_list.append(color_feature)
        processed_filepaths.append(filepath)
        # CNN 特徵稍後批次提取
        # 我們先保存 tensor 供批次處理
        # 注意：如果記憶體不足，這裡需要改成讀取圖片並立即提取，而不是保存所有 tensor
        # 為簡化，這裡假設記憶體足夠保存一批 tensor

    logger.info(
        f"圖像讀取和顏色特徵提取完成。成功: {len(processed_filepaths)} 張，失敗: {failed_count} 張。")
    if not processed_filepaths:
        logger.error("錯誤：未能成功處理任何圖像以進行備選聚類。")
        return None

    # 批次提取 ResNet CNN 特徵
    logger.info("開始批次提取 ResNet CNN 特徵...")
    temp_tensors = []  # 臨時儲存用於當前批次的 tensor
    temp_paths_indices = []  # 記錄 tensor 對應 processed_filepaths 的索引

    # 重新讀取已成功處理的圖片進行 CNN 預處理和特徵提取
    # 這裡可以優化：如果在上面循環中記憶體允許，可以保存 tensor
    # 但為確保邏輯清晰且應對大數據，這裡選擇重新讀取
    cnn_features_map = {}  # 使用字典儲存，key 為 filepath

    with torch.no_grad():
        tensors_for_batch = []
        paths_in_batch = []
        for filepath in processed_filepaths:
            filename = os.path.basename(filepath)
            img_bgr = cv2.imread(filepath)
            img_tensor = preprocess_image(
                img_bgr, resnet_inference_preprocess, filename)
            if img_tensor is not None:
                tensors_for_batch.append(img_tensor)
                paths_in_batch.append(filepath)

            # 當累積滿一個批次或處理到最後一個文件時，進行推理
            if len(tensors_for_batch) == BATCH_SIZE or filepath == processed_filepaths[-1]:
                if tensors_for_batch:
                    batch = torch.stack(tensors_for_batch).to(device)
                    logger.debug(f"  處理 ResNet 批次, 大小: {batch.shape[0]}")
                    batch_features = resnet_model(batch)
                    # 將特徵存入字典
                    for feature, path in zip(batch_features.cpu().numpy(), paths_in_batch):
                        cnn_features_map[path] = feature
                    tensors_for_batch = []  # 清空批次
                    paths_in_batch = []

    # 從字典中按 processed_filepaths 的順序提取 CNN 特徵
    cnn_features_list = [cnn_features_map[path]
                         for path in processed_filepaths if path in cnn_features_map]

    if len(cnn_features_list) != len(processed_filepaths):
        logger.error("錯誤：提取的 CNN 特徵數量與成功處理的文件數量不匹配。")
        # 可能需要更複雜的錯誤處理，比如找出哪些文件失敗了
        return None

    cnn_features_array = np.array(cnn_features_list)
    color_features_array = np.array(color_features_list)
    logger.info(
        f"成功提取 {len(cnn_features_array)} 個 ResNet CNN 特徵向量和 {len(color_features_array)} 個顏色特徵向量。")

    # 特徵縮放 (獨立縮放)
    scaler_cnn = StandardScaler()
    scaler_color = StandardScaler()
    scaled_cnn_features = scaler_cnn.fit_transform(cnn_features_array)
    scaled_color_features = scaler_color.fit_transform(color_features_array)
    logger.info("CNN 特徵和顏色特徵已分別進行標準化縮放。")

    # 合併特徵 (簡單拼接，可以考慮加權)
    # combined_features = np.hstack((scaled_cnn_features * CNN_FEATURE_WEIGHT_FALLBACK,
    #                              scaled_color_features * COLOR_FEATURE_WEIGHT_FALLBACK))
    combined_features = np.hstack((scaled_cnn_features, scaled_color_features))
    logger.info(f"合併後的特徵矩陣形狀: {combined_features.shape}")

    # 使用 Agglomerative Clustering 進行聚類
    logger.info(
        f"開始 Agglomerative Clustering (n_clusters={N_CLUSTERS}, linkage='{CLUSTER_LINKAGE}') 在合併的特徵上...")
    try:
        agg_clustering = AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            linkage=CLUSTER_LINKAGE
        )
        agg_clustering.fit(combined_features)
        labels = agg_clustering.labels_
        logger.info("備選方法聚類完成。")
        # 返回成功處理的文件路徑列表和對應的標籤
        return processed_filepaths, labels
    except Exception as e:
        logger.error(f"執行備選方法聚類時發生錯誤: {e}", exc_info=True)
        return None

# --- 主要聚類函數 (使用自訂模型) ---


def cluster_with_custom_model(image_paths):
    """使用自訂訓練的模型進行特徵提取和聚類"""
    global custom_model, device
    if custom_model is None or device is None:
        logger.error("自訂模型或設備未初始化。")
        return None

    logger.info("--- 開始使用 自訂訓練的 CNN 模型進行聚類 --- ")

    all_tensors = []
    processed_filepaths = []
    failed_count = 0

    # 1. 讀取、預處理圖像
    logger.info("開始讀取並預處理圖像 (為自訂模型準備)...")
    for filepath in image_paths:
        filename = os.path.basename(filepath)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            logger.warning(f"無法讀取圖片: {filepath}，跳過。")
            failed_count += 1
            continue

        img_tensor = preprocess_image(
            img_bgr, custom_inference_preprocess, filename)
        if img_tensor is None:
            logger.warning(f"跳過圖片 {filename}，因預處理失敗。")
            failed_count += 1
            continue

        all_tensors.append(img_tensor)
        processed_filepaths.append(filepath)

    logger.info(
        f"圖像預處理完成。成功: {len(processed_filepaths)} 張，失敗: {failed_count} 張。")
    if not all_tensors:
        logger.error("錯誤：未能成功預處理任何圖像。")
        return None

    # 2. 使用自訂模型批次提取特徵
    logger.info("開始使用自訂模型批次提取特徵...")
    all_features_list = []
    with torch.no_grad():
        for i in range(0, len(all_tensors), BATCH_SIZE):
            batch_tensors = all_tensors[i:i+BATCH_SIZE]
            batch = torch.stack(batch_tensors).to(device)
            logger.debug(
                f"  處理自訂模型批次 {i//BATCH_SIZE + 1}, 大小: {batch.shape[0]}")
            batch_features = custom_model(batch)
            all_features_list.extend(batch_features.cpu().numpy())

    if not all_features_list:
        logger.error("錯誤: 自訂模型特徵提取後列表為空。")
        return None

    features_array = np.array(all_features_list)
    logger.info(f"成功提取 {len(features_array)} 個自訂 CNN 特徵向量。")

    # 3. 特徵縮放
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
    logger.info("自訂 CNN 特徵已進行標準化縮放。")

    # 4. 使用 Agglomerative Clustering 進行聚類
    logger.info(
        f"開始 Agglomerative Clustering (n_clusters={N_CLUSTERS}, linkage='{CLUSTER_LINKAGE}') 在自訂特徵上...")
    try:
        agg_clustering = AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            linkage=CLUSTER_LINKAGE
        )
        agg_clustering.fit(scaled_features)
        labels = agg_clustering.labels_
        logger.info("自訂模型聚類完成。")
        return processed_filepaths, labels
    except Exception as e:
        logger.error(f"執行自訂模型聚類時發生錯誤: {e}", exc_info=True)
        return None

# --- 主程式邏輯 --- #


def main():
    global custom_model_loaded

    if not os.path.isdir(CAPTURE_DIR):
        logger.error(f"錯誤：找不到 capture 目錄 '{CAPTURE_DIR}'")
        return

    # 獲取 capture 目錄下的所有 .png 文件
    try:
        all_image_files = [os.path.join(CAPTURE_DIR, f) for f in os.listdir(
            CAPTURE_DIR) if f.lower().endswith('.png')]
    except OSError as e:
        logger.error(f"讀取 capture 目錄 '{CAPTURE_DIR}' 時出錯: {e}")
        return

    if not all_image_files:
        logger.info("capture 目錄下沒有找到 .png 圖片。")
        return

    logger.info(f"在 '{CAPTURE_DIR}' 中找到 {len(all_image_files)} 張 .png 圖片。")

    cluster_result = None
    if custom_model_loaded:
        # 優先使用自訂模型
        cluster_result = cluster_with_custom_model(all_image_files)
    else:
        # 使用備選方法
        cluster_result = cluster_with_resnet_color_features(all_image_files)

    if cluster_result is None:
        logger.error("聚類過程失敗，無法整理文件。")
        return

    # 解包結果
    processed_filepaths, labels = cluster_result

    # 獲取聚類結果統計
    unique_labels = set(labels)
    n_clusters_found = len(unique_labels)
    label_counts = {label: list(labels).count(label)
                    for label in unique_labels}

    logger.info(f"聚類完成。找到 {n_clusters_found} 個簇。")
    logger.info("每個簇的圖片數量:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  簇 {label}: {count} 張")

    # 5. 整理圖片到輸出目錄
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            logger.info(f"已刪除舊的輸出目錄: {OUTPUT_DIR}")
        except OSError as e:
            logger.error(f"刪除舊目錄 {OUTPUT_DIR} 失敗: {e}")
            # 不一定是致命錯誤，嘗試繼續創建
    try:
        os.makedirs(OUTPUT_DIR)
        logger.info(f"已創建新的輸出目錄: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"創建輸出目錄 {OUTPUT_DIR} 失敗: {e}")
        return  # 如果無法創建輸出目錄，則無法繼續

    logger.info("開始將圖片複製到對應的簇目錄...")
    copied_count = 0
    copy_failed_count = 0
    for i, filepath in enumerate(processed_filepaths):
        cluster_label = labels[i]
        # 創建簇子目錄 (如果不存在)
        cluster_subdir = os.path.join(
            OUTPUT_DIR, f"cluster_{cluster_label:02d}")
        try:
            if not os.path.exists(cluster_subdir):
                os.makedirs(cluster_subdir)
        except OSError as e:
            logger.error(
                f"創建簇子目錄 {cluster_subdir} 失敗: {e}，跳過文件 {os.path.basename(filepath)}")
            copy_failed_count += 1
            continue

        # 複製文件
        dest_path = os.path.join(cluster_subdir, os.path.basename(filepath))
        try:
            shutil.copy2(filepath, dest_path)  # copy2 保留元數據
            copied_count += 1
        except Exception as e:
            logger.error(f"複製文件 {filepath} 到 {dest_path} 失敗: {e}")
            copy_failed_count += 1

    logger.info(f"圖片整理完成。成功複製 {copied_count} 張圖片，失敗 {copy_failed_count} 次操作。")
    logger.info(f"聚類結果已保存在 '{OUTPUT_DIR}' 目錄下。")


if __name__ == "__main__":
    main()
