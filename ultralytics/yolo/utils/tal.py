# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checks import check_version
from .metrics import bbox_iou

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

######################################### Hàm này để lựa chọn những anchor nào nằm trong GT box #########################################
# Lấy tọa độ của anchor point trừ đi top-left, nếu < 0 => anchor point đó nằm ngoài gt box
# Lấy tọa độ của right-bottom trừ đi tọa độ anchor-point, nếu < 0 => anchor point nằm ngoài gt box
# Ứng với mỗi gt box (8) sẽ có các điểm anchor point (2100), xem các anchor point nào thuộc gt box đó
#########################################################################################################################################
def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]                     # (h*w, 2)          # (2100, 2)
    bs, n_boxes, _ = gt_bboxes.shape                    # (b, n_boxes, 4)   # (1, 8, 4)
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)       # left-top, right-bottom => lt = (8, 1, 2)

    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1) # (1, 8, 2100, 4)

    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    print("fg_mask:", fg_mask, fg_mask.size())
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        """mask_gt: (bs, số lượng bbox trong gt, 1) <=> (1, 8, 1)"""
        # để lựa chọn những anchor nào nằm trong GT box
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)  # (bs, n_boxes, h * w) (1, 8, 2100)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)

        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        """mask_gt: (bs, n_max_boxes, num_total_anchors) (1, 8, 2100) - mask tìm những anchor nằm trong các bbox của gt."""
        na = pd_bboxes.shape[-2]    # 2100
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
    
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)     # [1, 8, 2100]
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)  # [1, 8, 2100]
    
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # [2, 1, 8]

        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # [1, 8]
        ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj, [1, 8]

        # Get the scores of each grid for each gt cls

        # pd_scores có 2100 bbox với mỗi bbox sẽ có probabilities của 80 class [1, 2100, 80]
        # với 8 bbox của gt, mỗi bbox sẽ có 1 class tương ứng (ind [[0, 0, 0, 0, 0, 0, 0, 0], [22, 22, 22, 22, 22, 22, 22, 22]])
        # với mỗi bbox của gt, lấy ra predicted class probability của class tương ứng với gt trong các bbox predict ra [1, 8, 2100]
        # 8 gt, mỗi gt sẽ có 2100 bbox predict, trong mỗi bbox này chọn ra class probabilty của class tương ứng duy nhất.
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w     # 1, 8, 2100

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Trong mask_gt (1, 8, 2100), với 8 gt box ban đầu, mỗi box gt sẽ có 2100 anchor point cho gt box đó
        # Sau khi lọc thông qua select_candidates_in_gts(), trong 2100 anchor point đó sẽ chỉ có 1 vài anchor point nằm trong gt box
        # Tổng số lượng anchor point nằm trong toàn bộ 8 bbox ban đầu là: mask_gt.sum() = 156
        # vì vậy, khi pd_bboxes có size là (1, 2100, 4) tức là chỉ pred 2100 cho toàn bộ 8 bbox ban đầu, copy 2100 bbox này vào trong 8 gt tạo thành 1 list
        # mỗi gt sẽ chỉ chọn ra những anchor point nằm trong nó
        # tổng số lượng anchor point nằm trong toàn bộ 8 gt sẽ là: pd_boxes [156, 4] => 156 boxes
        pd_boxes = pd_bboxes.unsqueeze(1).repeat(1, self.n_max_boxes, 1, 1)[mask_gt]
        # Với mỗi gt_bboxes trog toàn bộ 8 gt, repeat mỗi box này 2100 lần rồi (tương ứng với 2100 predicted)
        # rồi loại bỏ những gt dựa vào mask như ở trên.
        gt_boxes = gt_bboxes.unsqueeze(2).repeat(1, 1, na, 1)[mask_gt]      # gt_boxes: torch.Size([156, 4])

        # Tính overlap của các bbox có anchor point nằm trong gt với các gt_boxes tương ứng 
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp(0)    # [1, 8, 2100]

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)                             

        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs[~topk_mask] = 0
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = torch.zeros(metrics.shape, dtype=torch.long, device=metrics.device)
        for it in range(self.topk):
            is_in_topk += F.one_hot(topk_idxs[:, :, it], num_anchors)
        # is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
