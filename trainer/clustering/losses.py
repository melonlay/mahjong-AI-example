import torch
import torch.nn as nn
import logging

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
