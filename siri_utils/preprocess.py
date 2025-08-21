import cv2
import torch
import numpy as np
from typing import List, Union


def to_int(obj):
    if isinstance(obj, tuple):
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = int(obj[i])
        return tuple(obj)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = int(obj[i])
        return obj
    else:
        assert False


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


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):

        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


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


def pre_transform(im, sz_wh):
    # sz_wh = 640, 640
    letterbox = LetterBox(
        tuple(reversed(sz_wh)),
        auto=False,
        scaleFill=False,
        scaleup=False,
        center=True
    )
    return [letterbox(image=x) for x in im]

def pre_transform_crop_left_right(im, ratio):
    return [crop_left_right(x, ratio/2, ratio/2) for x in im]

def pre_transform_crop(im):
    return [crop(x) for x in im]

def pre_transform_pad(im, sz, **kwargs):
    return [pad(x, to_sz_wh=sz, **kwargs) for x in im]

def preprocess(im: Union[np.ndarray, List[np.ndarray]], sz_wh, half=False) -> torch.Tensor: # to('cuda')
    assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
    
    im = np.stack(pre_transform(im, sz_wh))
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    # im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)

    im = im.to('cuda')
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(im: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(im, list):
        ret = []
        for image in im:
            assert isinstance(image, np.ndarray)
            ret.append(postprocess(image))
        return ret
    elif isinstance(im, np.ndarray):
        assert len(im.shape) == 3
        return im[..., ::-1]
    else:
        assert False


# def plot_image(image, post_process=True):
#     assert len(image.shape) == 3
#     if post_process:
#         if isinstance(image, torch.Tensor):
#             image.int()
#             image = image.cpu().numpy()
#             # image = image[::-1].transpose((1, 2, 0))
#         elif isinstance(image, np.ndarray):
#             # print(image.shape)
#             image.astype(np.uint8)
#             # image = image[..., ::-1]
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     assert isinstance(image, np.ndarray)
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def plot_yolo_results(results):
#     for index, result in enumerate(results):
#         result_image = result.plot()
#         result_image = result_image[..., ::-1]
#         plot_image(result_image)
#         # cv2.imwrite(f"output_{index}.jpg", result_image)


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


def embed_image_in_black_bg(small_img, target_height):
    """
    将小图像嵌入到黑色背景中，使其高度与目标高度一致
    :param small_img: 小图像
    :param target_height: 目标高度
    :return: 嵌入后的图像
    """
    small_height, small_width = small_img.shape[:2]
    # 创建一个黑色背景
    black_bg = np.zeros((target_height, small_width, 3), dtype=np.uint8)
    # 计算嵌入位置
    start_y = (target_height - small_height) // 2

    # 检查小图像是否为单通道
    if len(small_img.shape) == 2:
        # 将单通道图像转换为三通道图像
        small_img = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)

    # 将小图像嵌入到黑色背景中
    black_bg[start_y:start_y + small_height, :] = small_img
    return black_bg


def combime_wl_ir(wl_frame, ir_frame):
    wl_height, wl_width = wl_frame.shape[:2]
    ir_height, ir_width = ir_frame.shape[:2]
    if wl_height > ir_height:
        ir_frame = embed_image_in_black_bg(ir_frame, wl_height)
    else:
        wl_frame = embed_image_in_black_bg(wl_frame, ir_height)
    combined_frame = np.hstack((wl_frame, ir_frame))
    MAX_WIDTH = 1200
    if combined_frame.shape[1] > MAX_WIDTH:
        combined_frame = resize_image_to_width(combined_frame, MAX_WIDTH)
    return combined_frame


def _post_compute(fake_images):
    fake_images = torch.clamp(fake_images, -1., 1.)
    fake_images = (fake_images + 1) / 2.0  # [-1, 1] -> [0, 1]
    fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    fake_images = fake_images.astype(np.uint8)
    return fake_images



def combine_3vids(video1_path, video2_path, video3_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)
    
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        
        if not (ret1 and ret2 and ret3):
            break
            
        combined = cv2.hconcat([frame1, frame2, frame3])
        out.write(combined)
    
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()