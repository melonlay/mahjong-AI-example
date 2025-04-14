# tools/infer_and_organize.py
import sys
import os
import argparse
import logging
import shutil
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time

"""
使用訓練好的分類模型對指定目錄中的圖像進行推論，並將圖像根據預測結果整理到不同的子目錄中。

功能:
1.  接收命令行參數，包括模型路徑 (`--model_path`)、輸入圖像目錄 (`--input_dir`)、
    輸出目錄 (`--output_dir`) 以及其他推論相關參數 (批次大小、工作線程數等)。
2.  從指定的模型檢查點文件 (`.pth`) 加載訓練好的分類模型 (SimpleMahjongCNN)。
    - 會自動處理權重鍵名中可能存在的 'module.' 前綴。
    - 從檢查點中讀取類別名稱 (`class_names`) 和輸入尺寸 (`input_size`)。
3.  創建一個自訂的 `ImageFolderForInference` Dataset，用於從輸入目錄加載圖像。
4.  使用 DataLoader 進行批次加載和預處理（使用與驗證時相同的轉換）。
5.  對每個批次的圖像執行模型推論，獲得預測的類別索引。
6.  將預測索引轉換為類別名稱。
7.  清空並重新創建輸出目錄。
8.  在輸出目錄下，為每個預測出的類別創建子目錄。
9.  將輸入目錄中的每個原始圖像複製到輸出目錄下對應的預測類別子目錄中。
10. 記錄處理的圖片數量和花費的時間。

用法:
作為一個命令行工具直接運行。
```bash
# 假設模型保存在 trainer/classification/results/models/best_model.pth
# 待推論的圖片在 ./capture 目錄
# 結果輸出到 ./inference_output 目錄
python tools/infer_and_organize.py \
    --model_path trainer/classification/results/models/best_model.pth \
    --input_dir ./capture \
    --output_dir ./inference_output \
    --batch_size 64 \
    --num_workers 4
```

注意:
- `--input_dir` 中的圖像文件應該直接放在該目錄下，而不是放在子目錄中。
- `--output_dir` 如果已存在，會被清空。
- 需要 `trainer.classification.model` 和 `trainer.classification.dataset` 模組可用。
"""

# <<< 新增: 將專案根目錄添加到 sys.path >>>
# 假設此腳本位於 <project_root>/tools/
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # 使用 print 或 logging 取決於 logging 何時配置，這裡暫用 print
    # print(f"[DEBUG] Added project root to sys.path: {project_root}")

# <<< 現在直接導入，移除 try-except >>>
try:
    from trainer.classification.model import SimpleMahjongCNN
    from trainer.classification.dataset import get_transforms
except ImportError as e:
    # 配置 logging 後才能使用 logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # 確保 logging 已配置
    print(f"導入模組時出錯，即使已添加 project_root 到 sys.path: {e}")
    print(f"目前的 sys.path: {sys.path}")
    exit(1)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ImageFolderForInference(Dataset):
    """Loads images from a flat directory for inference."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"輸入目錄不存在: {data_dir}")

        for fname in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, fname)
            if os.path.isfile(path) and fname.lower().endswith(allowed_extensions):
                self.image_paths.append(path)
        if not self.image_paths:
            raise ValueError(f"在目錄 {data_dir} 中未找到任何支持的圖片文件。")
        logging.info(f"在 {data_dir} 中找到 {len(self.image_paths)} 張圖片進行推論。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Load image using PIL (robust to different formats)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path  # Return transformed image and its original path
        except Exception as e:
            logging.error(f"加載或轉換圖片失敗: {img_path} - {e}", exc_info=False)
            # Return None or a placeholder to be potentially filtered by collate_fn if needed
            # For simplicity, we might just let it raise here or return something identifiable
            return None, img_path  # Indicate failure for this item

# Optional: Collate function to filter out failed items


def collate_fn_skip_failures(batch):
    # Filter out items where image loading/transform failed (img is None)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None  # Return None if the whole batch failed
    # Use default collate for the rest
    return torch.utils.data.dataloader.default_collate(batch)


def infer_and_organize(args):
    """Loads model, performs inference, and organizes files."""
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"使用設備: {device}")

    # --- 加載模型 ---
    if not os.path.isfile(args.model_path):
        logging.error(f"模型檢查點未找到: {args.model_path}")
        return

    logging.info(f"正在從以下路徑加載檢查點: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # 確定模型參數 (從 checkpoint 或參數獲取)
    num_classes = None
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        logging.info(f"從檢查點加載類別名稱 ({num_classes} 個): {class_names}")
    else:
        logging.error("錯誤: 模型檢查點中未找到 'class_names'。無法將預測索引映射到名稱。")
        return

    input_size = checkpoint.get(
        'input_size', args.input_size)  # 從 checkpoint 或參數獲取
    if input_size != args.input_size:
        logging.warning(
            f"使用的輸入尺寸 ({input_size}) 與命令行參數 ({args.input_size}) 不同。將使用檢查點中的值。")

    # 初始化模型結構
    model = SimpleMahjongCNN(num_classes=num_classes,
                             input_size=input_size).to(device)

    # 加載權重
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        # 處理可能的 'module.' 前綴
        model_state_dict = {
            k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        logging.info("成功從 checkpoint['model_state_dict'] 加載模型權重。")
    else:
        logging.warning("檢查點不包含 'model_state_dict'。假設檢查點本身就是 state_dict。")
        model_state_dict = {
            k.replace('module.', ''): v for k, v in checkpoint.items()}
        try:
            model.load_state_dict(model_state_dict)
            logging.info("成功從 checkpoint 根加載模型權重。")
        except RuntimeError as e:
            logging.error(f"直接從 checkpoint 加載 state_dict 失敗: {e}")
            return

    model.eval()  # 設置為評估模式

    # --- 準備數據加載器 ---
    # 使用與驗證時相同的轉換 (不進行數據增強)
    inference_transform = get_transforms(input_size=input_size, augment=False)
    try:
        inference_dataset = ImageFolderForInference(
            args.input_dir, transform=inference_transform)
        inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers,
                                      collate_fn=collate_fn_skip_failures,  # 使用 collate_fn 過濾錯誤
                                      pin_memory=True)
    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"創建推論數據加載器時失敗: {e}")
        return

    # --- 準備輸出目錄 ---
    if os.path.exists(args.output_dir):
        logging.warning(f"輸出目錄 {args.output_dir} 已存在。將清空並重新創建。")
        try:
            shutil.rmtree(args.output_dir)
        except OSError as e:
            logging.error(f"清空輸出目錄時出錯: {e}")
            return
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"輸出目錄已創建: {args.output_dir}")
    except OSError as e:
        logging.error(f"創建輸出目錄時出錯: {e}")
        return

    # --- 執行推論和文件整理 ---
    logging.info("開始執行推論並整理文件...")
    processed_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_images, batch_paths in inference_loader:
            if batch_images is None:  # 如果整個批次都失敗了
                logging.warning("跳過一個空的批次 (所有圖片加載/轉換失敗)。")
                continue

            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted_indices = torch.max(outputs, 1)

            # 將預測結果整理到對應文件夾
            for i in range(len(batch_paths)):
                original_path = batch_paths[i]
                predicted_index = predicted_indices[i].item()
                predicted_class_name = class_names[predicted_index]

                # 創建目標子目錄
                target_subdir = os.path.join(
                    args.output_dir, predicted_class_name)
                os.makedirs(target_subdir, exist_ok=True)

                # 複製文件
                try:
                    target_path = os.path.join(
                        target_subdir, os.path.basename(original_path))
                    shutil.copy2(original_path, target_path)  # copy2 保留元數據
                    processed_count += 1
                except Exception as e:
                    logging.error(
                        f"複製文件 {original_path} 到 {target_subdir} 時出錯: {e}")

            if processed_count % 100 == 0 and processed_count > 0:
                logging.info(f"已處理 {processed_count} 張圖片...")

    end_time = time.time()
    logging.info(f"推論和文件整理完成。共處理 {processed_count} 張圖片。")
    logging.info(f"結果已保存到: {args.output_dir}")
    logging.info(f"總耗時: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mahjong Tile Inference and Organization')

    # 必要參數
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的分類模型檢查點路徑 (.pth file)')
    parser.add_argument('--input_dir', type=str, default='./capture',
                        help='包含待分類圖片的輸入目錄 (默認: ./capture)')
    parser.add_argument('--output_dir', type=str, default='./test_inference',
                        help='保存分類結果的輸出目錄 (默認: ./test_inference)')

    # 可選參數
    parser.add_argument('--input_size', type=int, default=96,
                        help='模型訓練時使用的輸入圖片尺寸 (用於加載模型和轉換)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='推論時的批次大小 (default: 64)')

    # 系統設置
    parser.add_argument('--num_workers', type=int, default=4,
                        help='數據加載線程數 (default: 4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='禁用 CUDA 推論')

    args = parser.parse_args()

    # 執行主函數
    infer_and_organize(args)
