"""
實現麻將牌特徵提取模型的訓練流程。

功能:
- 解析命令行參數 (例如數據路徑、學習率、批次大小等)。
- 設置隨機種子、設備 (CPU/GPU)。
- 創建數據加載器 (DataLoader)，可能使用自訂的 Dataset (來自 dataset.py) 和數據增強。
- 實例化模型 (來自 model.py) 和優化器 (例如 Adam)。
- 實現訓練循環 (training loop)，計算損失 (例如 SupConLoss)。
- 可能包含驗證循環 (validation loop)，計算評估指標 (例如 ARI, NMI)。
- 定期保存模型檢查點 (checkpoint) 和最佳模型。
- 使用 TensorBoard 或類似工具進行日誌記錄。

用法:
從命令行運行以開始訓練:
  python trainer/clustering/train.py --data_dir ./data --output_dir ./trainer/clustering [其他參數...]
"""

import os
import argparse
import logging
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# 導入 AMP
from torch.cuda.amp import GradScaler, autocast

# 從同目錄下的 model.py 導入模型
from model import SupConResNet

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Supervised Contrastive Loss (基本實現) ---


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
       (基本實現，可能缺少一些邊界情況處理或優化)
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """計算 SupCon loss
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 如果沒有標籤和 mask，則退化為自監督對比學習 (SimCLR)
            # 但在此場景下我們期望有標籤
            logger.warning("未提供 labels 或 mask 給 SupConLoss，將使用自對比 mask！")
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(
            features, dim=1), dim=0)  # (bsz * n_views, dim)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # (bsz, dim)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # (bsz * n_views, dim)
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits: (anchor_count * bsz, contrast_count * bsz)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask: (anchor_count * bsz, contrast_count * bsz)
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases (對角線元素)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # 保留非對角線的相同類別樣本

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 分母只計算非自身的樣本
        # 加 epsilon 防 log(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive samples
        mask_sum = mask.sum(1)
        # 避免除以零 (如果某個樣本沒有正樣本對，雖然理論上不應發生在 SupCon)
        valid_mask_indices = torch.where(mask_sum > 0)[0]

        if valid_mask_indices.shape[0] == 0:
            # 如果整個批次都沒有正樣本對 (例如 batch size 太小或類別太少且抽樣不均)
            logger.warning("此批次中沒有有效的正樣本對！ Loss 將為 0。")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 只計算有正樣本的行的 loss
        mean_log_prob_pos = (
            mask[valid_mask_indices] * log_prob[valid_mask_indices]).sum(1) / mask_sum[valid_mask_indices]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()  # 對批次中的有效樣本取平均

        return loss


# --- 帶雙視圖輸出的 Dataset ---
class MahjongContrastiveDataset(datasets.ImageFolder):
    """繼承 ImageFolder，但在 __getitem__ 中返回兩個視圖和標籤"""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)  # 在內部調用 self.transform
        if transform is None:
            raise ValueError("需要提供轉換函數以生成雙視圖")
        # 將傳入的生成雙視圖的函數賦值給 self.transform
        self.transform = transform
        logger.info(f"從 {root} 發現 {len(self.classes)} 個類別: {self.classes}")
        if len(self.samples) == 0:
            logger.warning(f"警告：在 {root} 中未找到任何圖像樣本！")

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            # 應用轉換函數，該函數應返回兩個視圖
            view1, view2 = self.transform(sample)
            return view1, view2, target
        except Exception as e:
            logger.error(f"加載或轉換圖像失敗: {path} - {e}", exc_info=True)
            # 返回 None 或一個標識符，讓 DataLoader 的 collate_fn 處理
            # 或者簡單地返回一個假的數據點 (如果允許)
            # 為了安全，這裡返回 None，需要在 DataLoader 中配置 collate_fn 或過濾
            # 更好的方法是在初始化時就檢查所有圖片
            # 暫時返回錯誤標籤，但這不好
            logger.warning(f"返回錯誤標籤 {-1} 以跳過樣本 {path}")
            dummy_tensor = torch.zeros((3, 96, 96))  # 假設尺寸是 96x96
            return dummy_tensor, dummy_tensor, -1  # 返回錯誤標籤

# --- 數據增強 (實際應用) ---

# 定義一個可 pickle 的轉換類


class TwoCropTransform:
    """為一個輸入圖像創建兩個隨機增強的視圖 (crops)"""

    def __init__(self, base_transform):
        """ Args: base_transform (callable): 應用於每個視圖的轉換 """
        self.base_transform = base_transform

    def __call__(self, x):
        # 對同一個輸入 x 應用兩次轉換，得到兩個不同的視圖
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2


def get_contrastive_transforms(size=96):
    """返回一個 TwoCropTransform 實例，包含隨機增強流程"""
    # ImageNet 均值和標準差
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 增強流程 (移除顏色抖動和灰度化)
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Removed ColorJitter
        # transforms.RandomApply([
        #     transforms.ColorJitter(
        #         brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1)
        # ], p=0.8),
        # Removed RandomGrayscale
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),  # Convert to tensor first
        normalize,  # Then normalize
        # Keep RandomErasing
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(
            0.3, 3.3), value='random', inplace=False)
    ])

    # 返回 TwoCropTransform 的實例，而不是 lambda
    return TwoCropTransform(augmentation)


# --- 處理損壞樣本的 Collate Function (移至頂層) ---
def collate_fn_filter_corrupt(batch):
    """過濾掉標籤為 -1 的壞樣本"""
    # 過濾掉標籤為 -1 的壞樣本
    batch = list(filter(lambda x: x[2] != -1, batch))
    if not batch:
        return None  # 如果整個批次都壞了，DataLoader 會處理這種情況（通常是跳過）
    # 使用 PyTorch 默認的 collate function 處理剩餘的有效樣本
    return torch.utils.data.dataloader.default_collate(batch)


# --- 訓練函數 ---
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, num_epochs):
    model.train()  # 確保模型處於訓練模式
    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()
    for i, batch_data in enumerate(dataloader):
        # 處理可能的 None batch (如果 collate_fn 返回 None)
        if batch_data is None:
            logger.warning(f"批次 {i+1} 為空 (所有樣本均損壞?)，跳過此批次。")
            continue
        view1, view2, labels = batch_data

        # # 過濾掉 __getitem__ 中可能返回的錯誤樣本 (已由 collate_fn 處理)
        # valid_indices = labels != -1
        # if not valid_indices.any():
        #     logger.warning(f"批次 {i+1} 中所有樣本均加載失敗，跳過此批次。")
        #     continue
        # view1 = view1[valid_indices].to(device)
        # view2 = view2[valid_indices].to(device)
        # labels = labels[valid_indices].to(device)
        view1, view2, labels = view1.to(
            device), view2.to(device), labels.to(device)

        # 使用 autocast 上下文
        with autocast():
            # 先通過模型獲取嵌入
            emb1 = model(view1)
            emb2 = model(view2)
            features = torch.cat([emb1.unsqueeze(1), emb2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)

        # 如果 loss 不是有效的數值，則跳過反向傳播
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}]: 計算出的 Loss 無效 ({loss.item()})，跳過此批次更新。")
            # 重置梯度，以防萬一之前的迭代有問題
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        # 使用 scaler 進行反向傳播
        scaler.scale(loss).backward()
        # 使用 scaler 更新優化器
        scaler.step(optimizer)
        # 更新 scaler
        scaler.update()

        total_loss += loss.item()
        valid_batches += 1

        if (i + 1) % 50 == 0:  # 每 50 個 batch 打印一次日誌
            elapsed_time = time.time() - start_time
            batches_done = i + 1
            batches_left = len(dataloader) - batches_done
            time_per_batch = elapsed_time / batches_done if batches_done > 0 else 0
            eta_seconds = batches_left * time_per_batch
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            current_avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batches_done}/{len(dataloader)}], '
                        f'Loss: {loss.item():.4f} (Avg: {current_avg_loss:.4f}), ETA: {eta_str}')

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    epoch_time = time.time() - start_time
    logger.info(
        f'Epoch [{epoch+1}/{num_epochs}] 完成, 平均 Loss: {avg_loss:.4f}, 耗時: {epoch_time:.2f} 秒')
    return avg_loss

# --- 新增：評估用數據轉換 ---


def get_eval_transforms(size=96):
    """返回用於評估的標準轉換 (無數據增強)"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])

# --- 新增：評估函數 ---


def evaluate_clustering(model_encoder, dataloader, device, num_classes, seed):
    """評估特徵提取器在聚類任務上的表現"""
    model_encoder.eval()  # 設置為評估模式
    all_features = []
    all_labels = []
    logger.info("開始提取評估集特徵...")
    with torch.no_grad():
        for images, labels in dataloader:
            # 過濾壞樣本 (雖然評估集理論上應該是乾淨的，但以防萬一)
            valid_indices = labels != -1
            if not valid_indices.any():
                continue
            images = images[valid_indices].to(device)
            labels = labels[valid_indices]

            features = model_encoder(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_features:
        logger.error("評估時未能提取任何有效特徵！")
        return -1.0, -1.0  # 返回無效值

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logger.info(f"評估集特徵提取完成，特徵矩陣形狀: {all_features.shape}")

    # 執行 K-Means 聚類
    logger.info(f"在提取的特徵上執行 K-Means (k={num_classes})...")
    kmeans = KMeans(n_clusters=num_classes,
                    random_state=seed, n_init=10, verbose=0)
    cluster_preds = kmeans.fit_predict(all_features)

    # 計算評估指標
    ari = adjusted_rand_score(all_labels, cluster_preds)
    nmi = normalized_mutual_info_score(all_labels, cluster_preds)
    logger.info(f"聚類評估結果 - ARI: {ari:.4f}, NMI: {nmi:.4f}")

    model_encoder.train()  # 恢復訓練模式
    return ari, nmi

# --- 主程序 ---


def main(args):
    # --- 設備 ---
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    logger.info(f"使用設備: {device}")

    # --- 種子和確定性 ---
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.info(f"設置隨機種子: {args.seed}")

    # --- 數據準備 (包含劃分) ---
    logger.info(f"從 '{args.data_dir}' 加載數據...")
    if not os.path.isdir(args.data_dir):
        logger.error(f"數據目錄 '{args.data_dir}' 不存在！")
        return

    # 創建兩個 transform
    contrastive_transform = get_contrastive_transforms(size=args.image_size)
    eval_transform = get_eval_transforms(size=args.image_size)

    # 創建兩個 Dataset 實例 (即使 root 相同，因為 transform 不同)
    # 訓練數據集需要返回雙視圖
    train_dataset_full = MahjongContrastiveDataset(
        root=args.data_dir, transform=contrastive_transform)
    # 評估數據集使用標準 ImageFolder 返回單個圖像
    eval_dataset_full = datasets.ImageFolder(
        root=args.data_dir, transform=eval_transform)

    if len(train_dataset_full.samples) == 0:
        logger.error(f"在 '{args.data_dir}' 中未找到任何圖像文件！請檢查路徑和文件夾結構。")
        return

    # 獲取標籤用於分層抽樣 (從任一數據集獲取即可，樣本順序應一致)
    targets = [s[1] for s in train_dataset_full.samples]
    num_classes = len(train_dataset_full.classes)
    logger.info(f"找到總樣本數: {len(targets)}, 總類別數: {num_classes}")

    # 劃分索引
    indices = list(range(len(targets)))
    try:
        train_indices, eval_indices = train_test_split(
            indices, test_size=args.eval_split, random_state=args.seed, stratify=targets)
        logger.info(
            f"數據集劃分完成。訓練集大小: {len(train_indices)}, 驗證集大小: {len(eval_indices)}")
    except ValueError as e:
        logger.error(f"數據劃分失敗 (可能某些類別樣本數過少無法分層): {e}")
        logger.warning("將使用所有數據進行訓練，不進行評估。")
        train_indices = indices
        eval_indices = []  # 不進行評估

    # 創建 Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    if eval_indices:
        eval_dataset = Subset(eval_dataset_full, eval_indices)
    else:
        eval_dataset = None

    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn_filter_corrupt)

    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size * 2,  # 評估時可以用稍大批次
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True
                                 # 移除 collate_fn，使用默認的即可
                                 # collate_fn=collate_fn_filter_corrupt
                                 )

    # --- 模型、損失函數、優化器 ---
    model = SupConResNet(
        name='resnet18', embedding_dim=args.embedding_dim).to(device)
    criterion = SupConLoss(temperature=args.temperature).to(device)

    # 推薦使用 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # (可選) 學習率調度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    scheduler = None  # 暫不使用

    # --- 初始化 GradScaler --- (僅當使用 CUDA 時)
    scaler = GradScaler(enabled=device.type == 'cuda')
    logger.info(f"啟用 GradScaler: {scaler.is_enabled()}")

    # --- 訓練循環 (加入評估和最佳模型保存) ---
    logger.info("開始訓練...")
    start_train_time = time.time()
    # <<< 初始化 top K 列表 >>>
    top_k_models = []  # 存儲 (ari_score, epoch, path)
    # <<< 移除舊的 best_ari, best_epoch 初始化 >>>
    # best_ari = -1.0
    # best_epoch = -1

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs)

        if scheduler:
            scheduler.step()

        # --- 執行評估 --- #
        current_ari = -1.0
        current_nmi = -1.0
        # <<< 修改條件以檢查 save_top_k > 0 >>>
        if eval_loader and args.save_top_k > 0 and \
           ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1):
            logger.info(f"--- Epoch {epoch+1} 結束，開始評估聚類效果 ---")
            current_ari, current_nmi = evaluate_clustering(
                model.encoder, eval_loader, device, num_classes, args.seed)
            logger.info(
                f"--- 評估完成 - ARI: {current_ari:.4f}, NMI: {current_nmi:.4f} ---")

            # --- 保存 Top K 模型 (基於 ARI) --- #
            save_dir = os.path.join(args.output_dir, 'result')
            os.makedirs(save_dir, exist_ok=True)

            # <<< 新增: 檢查是否需要保存 >>>
            should_save = len(top_k_models) < args.save_top_k or \
                (top_k_models and current_ari > top_k_models[0][0])

            if should_save:
                current_model_filename = f'model_epoch_{epoch+1}_ari_{current_ari:.4f}.pth'
                current_model_path = os.path.join(
                    save_dir, current_model_filename)
                logger.info(
                    f"*** 驗證集 ARI {current_ari:.4f} 達到 Top {args.save_top_k} 標準。正在保存模型到: {current_model_path} ***")
                try:
                    torch.save(model.encoder.state_dict(), current_model_path)
                    top_k_models.append(
                        (current_ari, epoch + 1, current_model_path))
                    top_k_models.sort(key=lambda x: x[0])  # 按 ARI 升序排序

                    if len(top_k_models) > args.save_top_k:
                        worst_model_info = top_k_models.pop(0)
                        worst_model_path = worst_model_info[2]
                        if os.path.exists(worst_model_path):
                            try:
                                os.remove(worst_model_path)
                                logger.info(
                                    f"已移除表現較差的模型文件: {os.path.basename(worst_model_path)}")
                            except OSError as e:
                                logger.error(
                                    f"移除舊模型文件 {worst_model_path} 時出錯: {e}")
                except Exception as e:
                    logger.error(
                        f"保存模型到 {current_model_path} 時失敗: {e}", exc_info=True)

            # <<< 新增: 打印當前 Top K 信息 >>>
            if top_k_models:
                top_scores_str = ", ".join([f"Epoch {e} (ARI:{s:.4f})" for s, e, p in sorted(
                    top_k_models, key=lambda x: x[0], reverse=True)])
                logger.info(f"當前 Top {len(top_k_models)} 模型: {top_scores_str}")

        # --- 定期保存檢查點 --- (邏輯不變)
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_dir = os.path.join(args.output_dir, 'result')
            os.makedirs(save_dir, exist_ok=True)
            feature_extractor_state = model.encoder.state_dict()
            save_path = os.path.join(
                save_dir, f'mahjong_feature_extractor_epoch_{epoch+1}.pth')
            try:
                torch.save(feature_extractor_state, save_path)
                logger.info(f'特徵提取器模型已保存到: {save_path}')
            except Exception as e:
                logger.error(f"保存模型到 {save_path} 時失敗: {e}", exc_info=True)

    total_train_time = time.time() - start_train_time
    logger.info(
        f"訓練完成！總耗時: {time.strftime('%H:%M:%S', time.gmtime(total_train_time))}")
    # <<< 修改結束日誌 >>>
    if top_k_models:
        logger.info("--- 最終 Top K 模型 (按 ARI 排序) ---")
        for ari, epoch_num, path in sorted(top_k_models, key=lambda x: x[0], reverse=True):
            logger.info(
                f"  Epoch {epoch_num}: ARI = {ari:.4f}, Path = {os.path.basename(path)}")  # 只顯示文件名
    # <<< 移除舊的 Best ARI 打印 >>>
    # if best_epoch != -1:
    #     logger.info(f"最佳模型出現在 Epoch {best_epoch}，對應的驗證集 ARI 為 {best_ari:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mahjong Tile CNN Training for Clustering (SupCon)')

    # 數據和模型路徑
    parser.add_argument('--data_dir', type=str,
                        default='./data', help='包含按類別分子目錄的圖像數據的路徑')
    parser.add_argument('--output_dir', type=str,
                        default='trainer/clustering', help='保存模型和日誌的輸出目錄')

    # 訓練超參數
    parser.add_argument('--epochs', type=int, default=100,
                        help='訓練輪數 (SupCon 可能需要較多輪數)')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='批次大小 (對比學習通常需要較大批次)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='學習率 (AdamW 常用的值)')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='權重衰減')
    parser.add_argument('--temperature', type=float,
                        default=0.1, help='SupConLoss 的溫度參數 (常見範圍 0.05-0.5)')
    parser.add_argument('--embedding_dim', type=int,
                        default=128, help='投影頭輸出的嵌入維度 (常見值 128, 256)')
    parser.add_argument('--image_size', type=int,
                        default=96, help='輸入圖像的尺寸 (正方形)')

    # 其他設置
    parser.add_argument('--num_workers', type=int,
                        default=4, help='數據加載線程數 (根據 CPU 核數調整)')
    parser.add_argument('--save_freq', type=int,
                        default=20, help='模型檢查點保存頻率 (epochs)')  # Checkpoint saving
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--cpu', action='store_true', help='強制使用 CPU')
    parser.add_argument('--eval_split', type=float,
                        default=0.15, help='驗證集劃分比例')
    parser.add_argument('--eval_freq', type=int,
                        default=5, help='評估頻率 (epochs)')
    # <<< 新增參數 >>>
    parser.add_argument('--save_top_k', type=int, default=1,
                        help='保存驗證集 ARI 表現最好的 K 個模型 (設為 0 則不根據 ARI 保存最佳模型)')

    args = parser.parse_args()

    # 打印參數
    logger.info("腳本參數:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # 創建輸出目錄
    os.makedirs(os.path.join(args.output_dir, 'result'), exist_ok=True)

    main(args)
