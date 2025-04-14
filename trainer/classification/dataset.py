# trainer/classification/dataset.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

"""
定義用於分類模型訓練、驗證和評估的數據集和數據加載器。

功能:
- `get_transforms(input_size, augment)`: 根據是否為訓練階段 (augment=True)
  返回包含圖像尺寸調整、數據增強 (可選) 和標準化的 PyTorch 轉換。
- `get_dataloaders(data_dir, input_size, batch_size, val_split, ...)`:
  從指定的 `data_dir` (應包含按類別命名的子目錄) 加載圖像數據。
  使用 `ImageFolder` 創建數據集。
  根據 `val_split` 比例將數據集劃分為訓練集和驗證集。
  為訓練集應用數據增強轉換，為驗證集應用標準轉換。
  返回訓練 DataLoader、驗證 DataLoader 和數據集中的類別名稱列表。
- `get_eval_loader(data_dir, input_size, batch_size, ...)`:
  從指定的 `data_dir` 加載圖像數據，創建一個用於評估的 DataLoader。
  只應用標準的 (非增強) 轉換。
  返回評估 DataLoader 和類別名稱列表。

用法:
主要由分類模型的訓練 (`trainer/classification/train.py`) 和評估 (`trainer/classification/eval.py`) 腳本導入。

- 在訓練腳本中:
  ```python
  from trainer.classification.dataset import get_dataloaders

  train_loader, val_loader, class_names = get_dataloaders(
      data_dir='./data', 
      input_size=96, 
      batch_size=128, 
      val_split=0.15
  )
  # ... 後續訓練循環 ...
  ```

- 在評估腳本中:
  ```python
  from trainer.classification.dataset import get_eval_loader

  eval_loader, class_names = get_eval_loader(
      data_dir='./path/to/test_data', 
      input_size=96, 
      batch_size=128
  )
  # ... 後續評估過程 ...
  ```
也可以直接運行此文件進行簡單的單元測試，它會嘗試從專案根目錄下的 `data` 目錄加載數據並打印信息:
```bash
python trainer/classification/dataset.py
```
"""

# Define image transformations
# Using ImageNet mean and std is a common starting point, even without pretraining
# Alternatively, calculate mean/std from your specific dataset later for better results.
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# Or simpler normalization:
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


def get_transforms(input_size=96, augment=True):
    """Get PyTorch transforms for training or validation."""
    transform_list = [
        transforms.Resize((input_size, input_size)),
    ]
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Optional
            # transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)), # Optional
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms.Compose(transform_list)


def get_dataloaders(data_dir, input_size=96, batch_size=32, val_split=0.2, num_workers=4, seed=42):
    """Creates training and validation DataLoaders."""

    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create datasets with appropriate transforms
    train_transform = get_transforms(input_size=input_size, augment=True)
    val_transform = get_transforms(input_size=input_size, augment=False)

    # Load the full dataset using ImageFolder
    # Load once to get stats and indices
    full_dataset_no_transform = datasets.ImageFolder(data_dir)
    logging.info(
        f"Found {len(full_dataset_no_transform)} images in {len(full_dataset_no_transform.classes)} classes: {full_dataset_no_transform.classes}")

    if len(full_dataset_no_transform) == 0:
        logging.error(
            f"No images found in {data_dir}. Check the directory structure.")
        raise ValueError(f"No images found in {data_dir}")

    # Split dataset into training and validation sets
    total_len = len(full_dataset_no_transform)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    if train_len == 0 or val_len == 0:
        logging.error(
            f"Dataset split resulted in zero samples for train ({train_len}) or validation ({val_len}). Adjust val_split or check dataset size.")
        raise ValueError(
            "Dataset split resulted in zero samples for train or validation.")

    logging.info(
        f"Splitting dataset: {train_len} training samples, {val_len} validation samples.")
    # Use a generator for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Create datasets with specific transforms *before* splitting
    # This is generally simpler to manage than applying transforms post-Subset
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # Split using random_split which handles indices internally
    train_subset, val_subset_dummy = random_split(
        train_dataset, [train_len, val_len], generator=generator)
    # We need to ensure the validation set uses the *validation* transforms.
    # Re-split the validation dataset using the same generator state ensures same indices.
    val_subset_dummy, val_subset = random_split(
        val_dataset, [train_len, val_len], generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full_dataset_no_transform.classes  # Return class names


def get_eval_loader(data_dir, input_size=96, batch_size=32, num_workers=4):
    """Creates a DataLoader for evaluation on a specific dataset."""
    if not os.path.isdir(data_dir):
        logging.error(f"Evaluation data directory not found: {data_dir}")
        raise FileNotFoundError(
            f"Evaluation data directory not found: {data_dir}")

    eval_transform = get_transforms(input_size=input_size, augment=False)

    eval_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    logging.info(
        f"Loaded evaluation dataset from {data_dir}: {len(eval_dataset)} images in {len(eval_dataset.classes)} classes.")

    if len(eval_dataset) == 0:
        logging.error(f"No images found in evaluation directory: {data_dir}")
        raise ValueError(f"No images found in {data_dir}")

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return eval_loader, eval_dataset.classes


if __name__ == '__main__':
    # Example usage:
    # Assume this script is in trainer/classification,
    # and data is in the root directory's 'data' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_directory = os.path.join(project_root, 'data')

    input_s = 96
    batch_s = 64

    try:
        logging.info(f"Looking for data in: {data_directory}")
        logging.info("Testing DataLoader creation...")
        train_dl, val_dl, class_names = get_dataloaders(
            data_directory, input_size=input_s, batch_size=batch_s, val_split=0.2)
        logging.info(
            f"Successfully created DataLoaders. Number of classes: {len(class_names)}")
        logging.info(f"Class names: {class_names}")

        # Fetch one batch from each loader to test
        logging.info("Fetching one batch from train_loader...")
        train_images, train_labels = next(iter(train_dl))
        logging.info(
            f"Train batch - Images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
        # Should be approx [-1, 1] if using mean/std 0.5
        logging.info(
            f"Train batch - Image tensor range: [{train_images.min():.2f}, {train_images.max():.2f}]")

        logging.info("Fetching one batch from val_loader...")
        val_images, val_labels = next(iter(val_dl))
        logging.info(
            f"Validation batch - Images shape: {val_images.shape}, Labels shape: {val_labels.shape}")

        logging.info("DataLoader test completed.")

        # Test evaluation loader
        logging.info("\nTesting Evaluation DataLoader creation...")
        # Use the validation split data directory for testing eval loader for now
        # In practice, you might have a separate test set directory
        eval_data_dir_test = data_directory  # Replace with actual test dir if available
        eval_dl, eval_class_names = get_eval_loader(
            eval_data_dir_test, input_size=input_s, batch_size=batch_s)
        logging.info(
            f"Successfully created Evaluation DataLoader. Number of classes: {len(eval_class_names)}")
        logging.info(f"Eval Class names: {eval_class_names}")
        eval_images, eval_labels = next(iter(eval_dl))
        logging.info(
            f"Evaluation batch - Images shape: {eval_images.shape}, Labels shape: {eval_labels.shape}")
        logging.info("Evaluation DataLoader test completed.")

    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
