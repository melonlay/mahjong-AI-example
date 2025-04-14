# trainer/classification/eval.py
"""
評估已訓練好的麻將牌分類模型的腳本。

功能:
1.  接收命令行參數，包括模型檢查點路徑 (`--model_path`)、評估數據目錄 (`--data_dir`)，
    以及可選的輸入尺寸、批次大小、類別數量等。
2.  從指定路徑加載模型檢查點 (.pth)。
    - 會嘗試從檢查點中讀取類別數量 (`class_names`)，如果沒有則需要用戶通過 `--num_classes` 指定，
      或者嘗試從數據目錄推斷。
    - 處理權重字典中可能的 `module.` 前綴。
3.  創建一個用於評估的 DataLoader (`get_eval_loader` from `dataset.py`)，加載評估數據集。
4.  初始化模型結構 (`SimpleMahjongCNN` from `model.py`) 並加載權重。
5.  定義損失函數 (CrossEntropyLoss)。
6.  調用 `evaluate_model` 函數在評估數據集上執行模型推論。
7.  `evaluate_model` 函數計算並記錄:
    - 平均損失 (Loss)
    - 準確率 (Accuracy)
    - 推論吞吐量 (samples/sec)
8.  如果 `evaluate_model` 在訓練循環中被調用 (提供了 `epoch` 和 `writer`)，
    它還可以將驗證損失和準確率寫入 TensorBoard。

用法:
作為一個命令行工具直接運行，用於評估一個已保存的模型檢查點。
```bash
# 假設模型保存在 trainer/classification/results/models/best_model.pth
# 評估數據在 ./test_data 目錄 (包含類別子目錄)
python trainer/classification/eval.py \
    --model_path trainer/classification/results/models/best_model.pth \
    --data_dir ./test_data \
    --batch_size 128 \
    --input_size 96 \
    --num_workers 4
```
也可以被訓練腳本 (`train.py`) 導入，用於在訓練過程中定期評估模型在驗證集上的性能。
```python
# 在 train.py 中:
# from trainer.classification.eval import evaluate_model
# ...
# val_loss, val_acc = evaluate_model(model, device, val_loader, criterion, epoch, writer)
```
"""
import os
import argparse
import logging
import time
import torch
import torch.nn as nn

# Import from local modules within the same package
# Use try-except for compatibility if run directly or as module
try:
    from .model import SimpleMahjongCNN
    from .dataset import get_eval_loader
except ImportError:
    # Fallback for running the script directly
    from model import SimpleMahjongCNN
    from dataset import get_eval_loader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_model(model, device, eval_loader, criterion, epoch=None, writer=None):
    """Evaluates the model on the given data loader."""
    model.eval()
    eval_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            eval_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

    eval_loss /= total_samples
    eval_acc = correct_predictions / total_samples
    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0

    # 構造日誌訊息，根據是否有 epoch 決定前綴
    log_prefix = f'--- Validation Epoch {epoch}' if epoch is not None else '--- Evaluation Complete'
    logging.info(f'{log_prefix} ---')
    logging.info(f'Avg Loss: {eval_loss:.4f}')
    logging.info(
        f'Accuracy: {eval_acc:.4f} ({correct_predictions}/{total_samples})')
    logging.info(f'Throughput: {throughput:.2f} samples/sec')

    # 如果提供了 writer 和 epoch，則寫入 TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/validation', eval_loss, epoch)
        writer.add_scalar('Accuracy/validation', eval_acc, epoch)

    return eval_loss, eval_acc


def main(args):
    """Main evaluation function."""
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.isfile(args.model_path):
        logging.error(f"Model checkpoint not found at: {args.model_path}")
        return

    logging.info(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Determine model parameters (num_classes, input_size)
    # Best practice: Save these in the checkpoint during training
    # Fallback: Infer or require as arguments
    num_classes = None
    if 'class_names' in checkpoint:
        num_classes = len(checkpoint['class_names'])
        logging.info(
            f"Number of classes loaded from checkpoint: {num_classes}")
    elif args.num_classes:
        num_classes = args.num_classes
        logging.info(
            f"Number of classes specified via argument: {num_classes}")

    # Input size should ideally match the training setting
    input_size = args.input_size
    logging.info(f"Using input size: {input_size}")

    # Load data *before* finalizing num_classes if needed
    logging.info(f"Loading evaluation data from: {args.data_dir}")
    try:
        eval_loader, loaded_class_names = get_eval_loader(
            args.data_dir,
            input_size=input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if num_classes is None:
            num_classes = len(loaded_class_names)
            logging.info(
                f"Inferred number of classes from data: {num_classes}")
        elif num_classes != len(loaded_class_names) and not args.num_classes:
            # Warn if checkpoint classes mismatch inferred classes, unless explicitly overridden
            logging.warning(f"Number of classes in checkpoint ({num_classes}) "
                            f"differs from data ({len(loaded_class_names)}). Using checkpoint value.")

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load evaluation data: {e}")
        return

    if num_classes is None:
        logging.error(
            "Could not determine the number of classes. Please specify via --num_classes or ensure data directory is valid.")
        return

    # Initialize Model structure
    model = SimpleMahjongCNN(num_classes=num_classes,
                             input_size=input_size).to(device)

    # Load weights from checkpoint state_dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        # Handle potential issues with DataParallel/DDP prefixes
        # Remove `module.` prefix if it exists
        model_state_dict = {
            k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        logging.info(
            "Model weights loaded successfully from checkpoint['model_state_dict'].")
    else:
        # Assume the checkpoint *is* the state_dict if 'model_state_dict' key is missing
        logging.warning(
            "Checkpoint does not contain 'model_state_dict' key. Assuming the checkpoint *is* the state dictionary.")
        model_state_dict = {
            k.replace('module.', ''): v for k, v in checkpoint.items()}
        try:
            model.load_state_dict(model_state_dict)
            logging.info(
                "Model weights loaded successfully from checkpoint root.")
        except RuntimeError as e:
            logging.error(
                f"Failed to load state_dict directly from checkpoint: {e}")
            logging.error(
                "Please ensure the checkpoint file contains a valid model state dictionary or has a 'model_state_dict' key.")
            return

    # --- Perform Evaluation ---
    criterion = nn.CrossEntropyLoss()
    evaluate_model(model, device, eval_loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mahjong Tile Classification Evaluation')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the evaluation image data')

    # Optional arguments with defaults matching train.py or common use cases
    parser.add_argument('--input_size', type=int, default=96,
                        help='Input image size used during training (default: 96)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (optional, inferred if possible)')

    # System settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of subprocesses for data loading (default: 4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA evaluation')

    args = parser.parse_args()
    main(args)
