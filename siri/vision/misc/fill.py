import cv2
import numpy as np
def flood_fill_segment(frame, seed_point, tolerance=(20, 20, 20), fill_color=(0, 0, 0)):
    """
    使用 floodFill 按颜色相似性填充区域
    :param frame: 输入图像 (BGR)
    :param seed_point: 种子点 (x, y)
    :param tolerance: 颜色容差（0-255）
    :param fill_color: 填充颜色 (BGR)
    :return: 填充后的图像 + 掩码
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h+2, w+2), dtype=np.uint8)  # floodFill 需要 +2 的掩码
    
    # 执行 floodFill
    cv2.floodFill(
        frame, mask, 
        seedPoint=seed_point,
        newVal=fill_color,
        loDiff=tolerance,  # BGR 下限
        upDiff=tolerance,  # BGR 上限
        flags=cv2.FLOODFILL_FIXED_RANGE | 8  # 8 连通区域
    )
    
    # 提取掩码（去掉多余的 2 像素）
    mask = mask[1:-1, 1:-1]
    
    return frame, mask

def find_darkest_pixel(image, center_w, center_h, radius=25):

    # 计算搜索区域的边界
    h_start = max(0, center_h - radius)
    h_end = min(image.shape[0], center_h + radius + 1)
    w_start = max(0, center_w - radius)
    w_end = min(image.shape[1], center_w + radius + 1)
    
    # 提取搜索区域
    search_region = image[h_start:h_end, w_start:w_end]
    
    if len(image.shape) == 3:
        search_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    
    # 找到最暗点的相对坐标
    min_val, _, min_loc, _ = cv2.minMaxLoc(search_region)
    
    # 转换为全局坐标
    min_h = h_start + min_loc[1]
    min_w = w_start + min_loc[0]
    
    return min_val, min_w, min_h