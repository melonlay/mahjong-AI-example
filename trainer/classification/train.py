# trainer/classification/train.py
"""
訓練麻將牌圖像分類模型的腳本。

功能:
1.  接收命令行參數，用於配置訓練過程，包括：
    - 數據目錄 (`--data_dir`)
    - 輸出目錄 (`--output_dir`)，用於保存模型和 TensorBoard 日誌
    - 訓練超參數：輪數 (`--epochs`)、批次大小 (`--batch_size`)、學習率 (`--lr`),
      權重衰減 (`--weight_decay`)、學習率衰減步長和因子 (`--lr_step_size`, `--lr_gamma`)
    - 驗證集比例 (`--val_split`)
    - 模型輸入尺寸 (`--input_size`)
    - 系統設置：工作線程數 (`--num_workers`)、隨機種子 (`--seed`)
    - 模型保存：保存最佳 K 個模型的數量 (`--save_top_k`)
    - 日誌記錄間隔 (`--log_interval`)
2.  設置隨機種子以保證可復現性。
3.  確定運行設備 (CPU 或 CUDA GPU)。
4.  創建輸出目錄 (用於模型和 TensorBoard 日誌)。
5.  初始化 TensorBoard 的 `SummaryWriter`。
6.  使用 `dataset.py` 中的 `get_dataloaders` 函數創建訓練和驗證 DataLoader。
7.  初始化分類模型 (`model.py` 中的 `SimpleMahjongCNN`)。
8.  初始化損失函數 (CrossEntropyLoss)、優化器 (AdamW) 和學習率調度器 (StepLR)。
9.  執行主訓練循環 (`args.epochs` 輪)：
    - 調用 `train_one_epoch` 函數訓練一個 epoch。
    - 調用 `eval.py` 中的 `evaluate_model` 函數在驗證集上評估模型。
    - 更新學習率調度器。
    - 記錄訓練和驗證的損失與準確率到 TensorBoard。
    - 根據驗證集準確率，保存排名前 `args.save_top_k` 的模型檢查點到模型目錄中。
      - 檢查點包含模型狀態、優化器狀態、輪數、類別名稱、輸入尺寸等信息。
      - 自動刪除不再是 Top K 的舊模型文件。
10. 訓練結束後，打印總耗時和最佳模型的驗證準確率。

用法:
作為一個命令行工具直接運行。
```bash
# 假設數據在 ./data 目錄，輸出到 trainer/classification/results 目錄
python trainer/classification/train.py \
    --data_dir ./data \
    --output_dir trainer/classification/results \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --input_size 96 \
    --save_top_k 3 \
    --num_workers 4 \
    --seed 42
```
訓練過程中可以使用 TensorBoard 查看損失和準確率曲線:
```bash
tensorboard --logdir trainer/classification/results/logs
```
"""
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
import torch
import time
import logging
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TF/oneDNN INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import from local modules within the same package
# Use try-except for compatibility if run directly or as module
try:
    from .model import SimpleMahjongCNN
    from .dataset import get_dataloaders
    from .eval import evaluate_model
except ImportError:
    # Fallback for running the script directly
    from model import SimpleMahjongCNN
    from dataset import get_dataloaders
    from eval import evaluate_model


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, writer, log_interval=50):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()
    batch_start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            current_time = time.time()
            elapsed_batch_time = current_time - batch_start_time
            batches_processed = log_interval
            throughput = batches_processed * train_loader.batch_size / \
                elapsed_batch_time if elapsed_batch_time > 0 else 0
            logging.info(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tThroughput: {throughput:.2f} samples/sec')
            batch_start_time = current_time  # Reset timer for next interval

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    logging.info(
        f'--- End Train Epoch {epoch}: Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} ---')
    return epoch_loss, epoch_acc


def main(args):
    """Main training function."""
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directory if it doesn't exist
    # Define output directories relative to the script location or project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir == 'results/classification':  # Default value check
        # Place results inside the trainer/classification folder if default
        output_base_dir = os.path.join(script_dir, 'results')
    else:
        # Use the specified path potentially relative to where script is called
        output_base_dir = args.output_dir

    log_dir = os.path.join(output_base_dir, 'logs',
                           time.strftime("%Y%m%d-%H%M%S"))
    model_dir = os.path.join(output_base_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f"Output directory: {output_base_dir}")
    logging.info(f"Log directory: {log_dir}")
    logging.info(f"Model directory: {model_dir}")

    writer = SummaryWriter(log_dir=log_dir)

    # Adjust data directory path if it's relative
    if args.data_dir == '../../data':  # Default value check
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(project_root, 'data')
    else:
        data_dir = args.data_dir

    # Get DataLoaders
    logging.info(f"Loading data from: {data_dir}")
    try:
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir,
            input_size=args.input_size,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load data: {e}")
        writer.close()
        return  # Exit if data loading fails

    num_classes = len(class_names)
    logging.info(f"Number of classes detected: {num_classes}")

    # Initialize Model
    model = SimpleMahjongCNN(num_classes=num_classes,
                             input_size=args.input_size).to(device)
    logging.info(f"Model initialized: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {num_params:,}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Initialize top K models list
    top_k_models = []

    # Training Loop
    start_time = time.time()
    logging.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch, writer, args.log_interval)
        val_loss, val_acc = evaluate_model(
            model, device, val_loader, criterion, epoch=epoch, writer=writer)
        scheduler.step()  # Step the scheduler

        # Save top K models based on validation accuracy
        if args.save_top_k > 0:
            should_save = len(top_k_models) < args.save_top_k or \
                (top_k_models and val_acc > top_k_models[0][0])

            if should_save:
                current_model_filename = f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth'
                current_model_path = os.path.join(
                    model_dir, current_model_filename)
                logging.info(
                    f"*** Validation accuracy {val_acc:.4f} meets Top {args.save_top_k} standard. Saving model to: {current_model_path} ***")
                try:
                    save_content = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_acc': val_acc,
                        'loss': val_loss,
                        'class_names': class_names,
                        'input_size': args.input_size
                    }
                    torch.save(save_content, current_model_path)

                    top_k_models.append((val_acc, epoch, current_model_path))
                    top_k_models.sort(key=lambda x: x[0])

                    if len(top_k_models) > args.save_top_k:
                        worst_model_info = top_k_models.pop(0)
                        worst_model_path = worst_model_info[2]
                        if os.path.exists(worst_model_path):
                            try:
                                os.remove(worst_model_path)
                                logging.info(
                                    f"Removed old model file: {os.path.basename(worst_model_path)}")
                            except OSError as e:
                                logging.error(
                                    f"Error removing old model file {worst_model_path}: {e}")
                except Exception as e:
                    logging.error(
                        f"Error saving model to {current_model_path}: {e}", exc_info=True)

            if top_k_models:
                top_scores_str = ", ".join([f"Epoch {e} (Acc:{s:.4f})" for s, e, p in sorted(
                    top_k_models, key=lambda x: x[0], reverse=True)])
                logging.info(
                    f"Current Top {len(top_k_models)} models (by accuracy): {top_scores_str}")

        # Save checkpoint periodically
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                model_dir, f'checkpoint_epoch_{epoch}.pth')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'last_val_acc': val_acc,
                    'loss': val_loss,
                    'class_names': class_names,
                    'input_size': args.input_size
                }, checkpoint_path)
                logging.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logging.error(
                    f"Error saving checkpoint to {checkpoint_path}: {e}", exc_info=True)

    total_training_time = time.time() - start_time
    logging.info(
        f"Training finished in {total_training_time / 3600:.2f} hours ({total_training_time:.2f} seconds).")
    if top_k_models:
        logging.info("--- Final Top K models (by validation accuracy) ---")
        for acc, epoch_num, path in sorted(top_k_models, key=lambda x: x[0], reverse=True):
            logging.info(
                f"  Epoch {epoch_num}: Accuracy = {acc:.4f}, Path = {os.path.basename(path)}")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mahjong Tile Classification Training')

    # Data and directories
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing the image data (default: ../../data relative to script)')
    parser.add_argument('--output_dir', type=str, default='results/classification',
                        help='Directory to save logs and models relative to script dir (default: results/classification)')
    parser.add_argument('--input_size', type=int, default=96,
                        help='Input image size (default: 96)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) (default: 1e-4)')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Step size for learning rate scheduler (default: 10)')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Factor to reduce learning rate by (default: 0.5)')

    # System settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of subprocesses to use for data loading (default: 4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed (default: 42)')

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50,
                        help='How many batches to wait before logging training status (default: 50)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='How many epochs to wait before saving a checkpoint (0 to disable) (default: 10)')
    parser.add_argument('--save_top_k', type=int, default=1,
                        help='Save top K models based on validation accuracy (0 to disable)')

    args = parser.parse_args()
    main(args)
