import cv2
import numpy as np

def apply_gamma(image, gamma):
    # 对比度提升（使用gamma校正）
    gamma = 0.4  # 小于1的值增加对比度
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_contrast(image, contrast_coef):
    """
    调整图像的对比度
    :param image: 输入图像 (H, W, C)
    :param contrast_coef: 对比度系数，越大对比度越高
    :return: 调整对比度后的图像
    """
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3
    
    # 将图像转换为浮点型以便计算
    image = image.astype(np.float32)
    
    # 计算图像的平均亮度
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    
    # 应用对比度调整公式: new_pixel = (old_pixel - mean) * contrast_coef + mean
    contrasted_image = (image - mean) * contrast_coef + mean
    
    contrasted_image = np.clip(contrasted_image, 0, 255)
    
    return contrasted_image.astype(np.uint8)