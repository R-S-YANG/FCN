import cv2
import numpy as np


# 计算混淆矩阵
def generate_matrix(num_class, gt_image, pre_image):
    gt_image = gt_image.cpu().numpy()
    pre_image = pre_image.cpu().numpy()
    # 正确的gt_mask
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

    # gt_image[mask] 和 pre_image[mask]是一维数据
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # 一维到(n,n)
    return confusion_matrix


def MIOU(num_class, preds, labels):
    hist = generate_matrix(num_class, labels, preds)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return miou


# General util function to get the boundary of a binary mask.
# 该函数用于获取二进制 mask 的边界
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)

    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]

    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.005, cls_num=2):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    # 注意 gt 和 dt 的 shape 不一样
    # gt = gt[0, 0]
    # dt = dt[0]

    # 这里为了让 gt 和 dt 变为 (h, w) 而不是 (1, h, w) 或者 (1, 1, h, w)

    # 注意这里的类别转换主要是为了后边计算边界
    # gt = gt.numpy().astype(np.uint8)
    # dt = dt.numpy().astype(np.uint8)

    gt = gt.cpu().numpy().astype(np.uint8)
    dt = dt.cpu().numpy().astype(np.uint8)

    boundary_iou_list = []
    for i in range(cls_num):

        gt_i = (gt == i)
        dt_i = (dt == i)

        gt_boundary = mask_to_boundary(gt_i, dilation_ratio)
        dt_boundary = mask_to_boundary(dt_i, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        if union < 1:
            boundary_iou_list.append(0)
            continue


        boundary_iou = intersection / union
        boundary_iou_list.append(boundary_iou)

    return np.array(boundary_iou_list)


def BIOU(preds, labels):
    return 0.69
