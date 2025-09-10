import cv2
import numpy as np
from uhtk.siri.utils.iterable_tools import to_int

###########################################################################################################
# crop
def crop_wh(frame, margin_pix_w, margin_pix_h):
    orig_shape = frame.shape
    assert 2*margin_pix_h < orig_shape[0]
    assert 2*margin_pix_w < orig_shape[1]
    frame = frame[margin_pix_h:orig_shape[0]-margin_pix_h, margin_pix_w:orig_shape[1]-margin_pix_w]
    return frame


def crop(image_origin):
    height, width = image_origin.shape[:2]

    if width > height:
        offset = (width - height) // 2
        image_cropped = image_origin[:, offset:offset + height]
    else:
        offset = (height - width) // 2
        image_cropped = image_origin[offset:offset + width, :]

    return image_cropped

def crop_left_right(x: np.ndarray, left_ratio: float, right_ratio: float):
    """
    对 numpy 图像进行左右裁剪，去除 width 指定比例的部分。

    :param x: 输入图像 (numpy 数组, 形状: HxWxC 或 HxW)
    :param left_ratio: 左侧裁剪的比例 (0.0 ~ 1.0)
    :param right_ratio: 右侧裁剪的比例 (0.0 ~ 1.0)
    :return: 裁剪后的图像 (numpy 数组)
    """
    assert 0.0 <= left_ratio < 1.0, "left_ratio 必须在 0.0 和 1.0 之间"
    assert 0.0 <= right_ratio < 1.0, "right_ratio 必须在 0.0 和 1.0 之间"
    assert left_ratio + right_ratio < 1.0, "裁剪比例之和必须小于 1.0"

    h, w = x.shape[:2]
    left_crop = int(w * left_ratio)
    right_crop = int(w * right_ratio)

    return x[:, left_crop:w - right_crop]




###########################################################################################################
# pad
def pad(image_origin, to_sz_wh=None, white_bg=False):
    if to_sz_wh is not None:
        to_size_w, to_size_h = to_sz_wh
        height, width = image_origin.shape[:2]

        if to_size_h < height and to_size_w < width:
            if to_size_h/height < to_size_w/width:
                image_origin = cv2.resize(image_origin, to_int((width * to_size_h/height, to_size_h,)))
            else:
                image_origin = cv2.resize(image_origin, to_int((to_size_w, height * to_size_w/width,)))
        elif to_size_h >= height and to_size_w >= width:
            if to_size_h/height < to_size_w/width:
                image_origin = cv2.resize(image_origin, to_int((width * to_size_h/height, to_size_h,)))
            else:
                image_origin = cv2.resize(image_origin, to_int((to_size_w, height * to_size_w/width,)))
        else:
            raise NotImplementedError
        # print(to_sz_wh, " ", width, " ", height)
    else:
        to_size_w = max(width, height);to_size_h = to_size_w

    height, width = image_origin.shape[:2]

    if white_bg:
        padded_image = np.ones((to_size_h, to_size_w, 3), dtype=image_origin.dtype) * 255
    else:
        padded_image = np.zeros((to_size_h, to_size_w, 3), dtype=image_origin.dtype)

    if to_size_h > height:
        y_offset = (to_size_h - height) // 2
        padded_image[y_offset:y_offset + height, :] = image_origin
    else:
        x_offset = (to_size_w - width) // 2
        padded_image[:, x_offset:x_offset + width] = image_origin

    padded_image.astype(np.uint8)
    return padded_image


###########################################################################################################
# resize
def resize_image_to_width(image, target_width):
    """
    将图片等比缩放到指定宽度
    :param image: 输入的图片（可以是通过 cv2.imread 读取的图片）
    :param target_width: 目标宽度
    :return: 缩放后的图片
    """
    # 获取原始图片的高度和宽度
    original_height, original_width = image.shape[:2]
    # 计算缩放比例
    scale_ratio = target_width / original_width
    # 计算缩放后的高度
    target_height = int(original_height * scale_ratio)
    # 使用 cv2.resize 函数进行缩放，插值方法使用 cv2.INTER_AREA 以保证质量
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image


