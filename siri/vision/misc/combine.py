import cv2
import numpy as np
from .crop_pad_resize import resize_image_to_width


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