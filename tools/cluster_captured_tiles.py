"""
對 `capture/` 目錄下的麻將牌圖像執行聚類。

主要功能:
1.  讀取 `capture/` 目錄中的所有圖像文件。
2.  優先嘗試從 `--model_path` 參數指定的路徑加載自訂訓練的聚類特徵提取模型。
3.  如果自訂模型加載成功，使用該模型提取圖像特徵。
4.  如果無法加載自訂模型（未提供路徑、文件不存在或加載失敗），則自動回退 (Fallback)
    到使用預訓練的 ResNet18 模型提取 CNN 特徵，並結合計算的 HSV 顏色直方圖作為補充特徵。
5.  對提取出的特徵向量進行標準化 (StandardScaler)。
6.  使用凝聚聚類 (Agglomerative Clustering) 算法對標準化後的特徵進行聚類。
7.  根據聚類結果，將 `capture/` 目錄下的原始圖像複製到 `clustered_tiles/` 目錄下
    對應的簇子目錄 (例如 `cluster_00`, `cluster_01`, ...) 中。

用法:
直接運行此腳本。可以選擇性地提供自訂模型路徑。

- 使用預設回退方法 (ResNet18 + Color):
  ```bash
  python tools/cluster_captured_tiles.py
  ```

- 使用自訂訓練的聚類模型:
  ```bash
  # 將 <path_to_your_model.pth> 替換為實際模型檔案路徑
  python tools/cluster_captured_tiles.py --model_path trainer/clustering/result/your_model.pth
  ```

(運行前請確保 `capture/` 目錄中有待聚類的圖片)

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
import argparse

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定 (部分移到 argparse) ---
CAPTURE_DIR = "capture"  # 包含擷取圖片的目錄
OUTPUT_DIR = "clustered_tiles"  # 存放分類結果的目錄

# --- 通用設定 ---
RESIZE_DIM_CNN = (96, 96)  # 與訓練時的 image_size 保持一致，也用於 ResNet18 Fallback
N_CLUSTERS = 30
CLUSTER_LINKAGE = 'ward'
BATCH_SIZE = 32  # 特徵提取批次大小

# --- 移除寫死的模型路徑常數 ---
# CUSTOM_MODEL_PATH = "trainer/clustering/result/best_mahjong_feature_extractor.pth"

# --- ResNet18 + 顏色特徵 Fallback 設定 ---
RESIZE_DIM_COLOR = (64, 64)  # Fallback 時計算顏色直方圖的大小
HSV_HIST_BINS = [8, 4, 4]  # H, S, V 的 bins 數量 (Fallback 使用)

# --- 全局設定 PyTorch 設備 ---
device = None
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch 將使用設備: {device}")
except Exception as e:
    logger.critical(f"設定 PyTorch 設備時失敗: {e}", exc_info=True)
    device = None  # 標記設備設定失敗

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
        return None, None

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
        return None, None

    cnn_features_list = []
    color_features_list = []
    processed_filepaths = []
    failed_count = 0
    resnet_preprocess = get_resnet_inference_transforms()

    logger.info("開始讀取圖像並提取 ResNet CNN 特徵和顏色特徵...")
    img_tensors_batch = []  # For batching CNN features

    for idx, filepath in enumerate(image_paths):
        filename = os.path.basename(filepath)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            logger.warning(f"無法讀取圖片: {filepath}，跳過。")
            failed_count += 1
            continue

        color_feature = calculate_hsv_histogram(
            img_bgr, HSV_HIST_BINS, RESIZE_DIM_COLOR, filename)
        if color_feature is None:
            logger.warning(f"跳過圖片 {filename}，因顏色特徵提取失敗。")
            failed_count += 1
            continue

        img_tensor = preprocess_image(img_bgr, resnet_preprocess, filename)
        if img_tensor is None:
            logger.warning(f"跳過圖片 {filename}，因 CNN 預處理失敗。")
            failed_count += 1
            continue

        color_features_list.append(color_feature)
        processed_filepaths.append(filepath)
        img_tensors_batch.append(img_tensor)

        if len(img_tensors_batch) == BATCH_SIZE or idx == len(image_paths) - 1:
            if img_tensors_batch:
                input_batch = torch.stack(img_tensors_batch).to(device)
                with torch.no_grad():
                    batch_features = resnet_model(input_batch)
                cnn_features_list.append(batch_features.cpu().numpy())
                img_tensors_batch = []

    logger.info(
        f"圖像讀取和特徵提取完成。成功: {len(processed_filepaths)} 張，失敗: {failed_count} 張。")
    if not processed_filepaths or not cnn_features_list:
        logger.error("錯誤：未能成功處理任何圖像以進行備選聚類。")
        return None, None

    all_cnn_features = np.vstack(cnn_features_list)
    all_color_features = np.array(color_features_list)
    logger.info(
        f"合併後 CNN 特徵形狀: {all_cnn_features.shape}, 顏色特徵形狀: {all_color_features.shape}")

    # 標準化
    logger.info("標準化兩種特徵...")
    scaler_cnn = StandardScaler()
    scaler_color = StandardScaler()
    all_cnn_features_scaled = scaler_cnn.fit_transform(all_cnn_features)
    all_color_features_scaled = scaler_color.fit_transform(all_color_features)

    # 合併特徵 (簡單拼接)
    all_features_combined = np.concatenate(
        (all_cnn_features_scaled, all_color_features_scaled), axis=1)
    logger.info(f"合併並標準化後總特徵形狀: {all_features_combined.shape}")

    # 執行聚類
    logger.info(
        f"使用 AgglomerativeClustering (n_clusters={N_CLUSTERS}, linkage='{CLUSTER_LINKAGE}') 進行聚類...")
    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage=CLUSTER_LINKAGE, metric='euclidean')
    labels = clustering.fit_predict(all_features_combined)
    logger.info("聚類完成。")

    return labels, processed_filepaths

# --- 使用自訂模型的聚類函數 --- #


def cluster_with_custom_model(image_paths, custom_model, preprocess_transform):
    global device
    if not custom_model or device is None:
        logger.error("未提供有效的自訂模型或 PyTorch 設備未初始化。")
        return None, None

    logger.info("--- 執行自訂模型聚類方法 ---")
    features_list = []
    processed_filepaths = []
    failed_count = 0
    img_tensors_batch = []

    logger.info("開始讀取圖像並準備批次進行自訂模型特徵提取...")
    for idx, filepath in enumerate(image_paths):
        filename = os.path.basename(filepath)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            logger.warning(f"無法讀取圖片: {filepath}，跳過。")
            failed_count += 1
            continue

        img_tensor = preprocess_image(img_bgr, preprocess_transform, filename)
        if img_tensor is None:
            logger.warning(f"跳過圖片 {filename}，因預處理失敗。")
            failed_count += 1
            continue

        img_tensors_batch.append(img_tensor)
        processed_filepaths.append(filepath)

        if len(img_tensors_batch) == BATCH_SIZE or idx == len(image_paths) - 1:
            if img_tensors_batch:
                input_batch = torch.stack(img_tensors_batch).to(device)
                with torch.no_grad():
                    batch_features = custom_model(input_batch)
                features_list.append(batch_features.cpu().numpy())
                img_tensors_batch = []

    logger.info(
        f"特徵提取完成。成功: {len(processed_filepaths)} 張，失敗: {failed_count} 張。")
    if not processed_filepaths or not features_list:
        logger.error("錯誤：未能成功處理任何圖像以提取特徵。")
        return None, None

    all_features = np.vstack(features_list)
    logger.info(f"合併後總特徵形狀: {all_features.shape}")

    logger.info("標準化提取的特徵...")
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    logger.info(
        f"使用 AgglomerativeClustering (n_clusters={N_CLUSTERS}, linkage='{CLUSTER_LINKAGE}') 進行聚類...")
    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage=CLUSTER_LINKAGE, metric='euclidean')
    labels = clustering.fit_predict(all_features_scaled)
    logger.info("聚類完成。")

    return labels, processed_filepaths


# --- 主函數 --- #
def main(args):
    global device
    if device is None:
        logger.critical("PyTorch 設備初始化失敗，無法繼續。")
        return

    # --- 決定使用哪個模型 ---
    custom_model_to_use = None
    preprocess_transform_to_use = None
    use_fallback = True

    if args.model_path:
        logger.info(f"嘗試從 '{args.model_path}' 加載自訂模型...")
        if not os.path.exists(args.model_path):
            logger.warning(f"提供的模型文件 '{args.model_path}' 不存在。")
        else:
            try:
                loaded_model = models.resnet18(weights=None)
                num_ftrs = loaded_model.fc.in_features
                loaded_model.fc = torch.nn.Identity()

                state_dict = torch.load(args.model_path, map_location='cpu')
                has_prefix = any(key.startswith('encoder.')
                                 for key in state_dict.keys())
                if has_prefix:
                    logger.info("檢測到 'encoder.' 前綴，將進行移除...")
                    new_state_dict = {k[len('encoder.'):] if k.startswith(
                        'encoder.') else k: v for k, v in state_dict.items()}
                    state_dict = new_state_dict

                model_keys = set(loaded_model.state_dict().keys())
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() if k in model_keys}
                missing_keys, unexpected_keys = loaded_model.load_state_dict(
                    filtered_state_dict, strict=False)

                if unexpected_keys:
                    logger.warning(f"加載狀態字典時發現未預期的鍵: {unexpected_keys}")
                if missing_keys:
                    logger.warning(f"加載狀態字典時缺少鍵: {missing_keys}")

                loaded_model.to(device)
                loaded_model.eval()
                custom_model_to_use = loaded_model
                preprocess_transform_to_use = get_custom_inference_transforms()
                use_fallback = False
                logger.info(f"成功加載自訂模型 '{args.model_path}'。")

            except Exception as e:
                logger.error(
                    f"加載自訂模型 '{args.model_path}' 時發生錯誤: {e}", exc_info=True)
                logger.warning("加載自訂模型失敗。")

    if use_fallback:
        logger.warning(f"未提供有效模型路徑或加載失敗。將使用備選方法 (ResNet18 + 顏色)。")

    # --- 讀取 capture 目錄下的圖片 ---
    if not os.path.isdir(CAPTURE_DIR):
        logger.error(f"錯誤：找不到 capture 目錄 '{CAPTURE_DIR}'")
        return
    image_files = [os.path.join(CAPTURE_DIR, f) for f in os.listdir(CAPTURE_DIR)
                   if os.path.isfile(os.path.join(CAPTURE_DIR, f)) and
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        logger.info(f"Capture 目錄 '{CAPTURE_DIR}' 中沒有找到圖片文件。")
        return
    logger.info(f"在 '{CAPTURE_DIR}' 中找到 {len(image_files)} 張圖片。")

    # --- 執行聚類 ---
    cluster_labels = None
    processed_filepaths = None

    if not use_fallback and custom_model_to_use:
        cluster_labels, processed_filepaths = cluster_with_custom_model(
            image_files, custom_model_to_use, preprocess_transform_to_use)
    else:
        cluster_labels, processed_filepaths = cluster_with_resnet_color_features(
            image_files)

    # --- 處理聚類結果 ---
    if cluster_labels is None or processed_filepaths is None:
        logger.error("聚類過程失敗，無法整理文件。")
        return

    if len(cluster_labels) != len(processed_filepaths):
        logger.error(
            f"錯誤：聚類標籤數量 ({len(cluster_labels)}) 與成功處理的文件數量 ({len(processed_filepaths)}) 不匹配！")
        return

    if os.path.exists(OUTPUT_DIR):
        logger.info(f"輸出目錄 '{OUTPUT_DIR}' 已存在，將清空...")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except OSError as e:
            logger.error(f"清空輸出目錄時出錯: {e}")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"創建輸出目錄 '{OUTPUT_DIR}' 失敗: {e}")
        return

    logger.info(
        f"開始將 {len(processed_filepaths)} 張圖片複製到 '{OUTPUT_DIR}' 下的聚類子目錄...")
    copied_count = 0
    copy_errors = 0
    for i, label in enumerate(cluster_labels):
        original_filepath = processed_filepaths[i]
        cluster_dir = os.path.join(OUTPUT_DIR, f"cluster_{label:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        try:
            shutil.copy2(original_filepath, cluster_dir)
            copied_count += 1
        except Exception as e:
            logger.error(f"複製文件 {original_filepath} 到 {cluster_dir} 時出錯: {e}")
            copy_errors += 1

    logger.info(f"文件複製完成。成功複製: {copied_count} 張，複製失敗: {copy_errors} 張。")
    logger.info(f"聚類結果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='對 capture/ 目錄中的麻將牌進行聚類')
    parser.add_argument('--model_path', type=str, default=None,
                        help='（可選）指定要使用的自訂聚類模型 (.pth) 的路徑。如果未提供或加載失敗，將使用備選方法。')
    args = parser.parse_args()
    main(args)
