import cv2
import torch
import numpy as np
from typing import List, Union
from misc.crop_pad_resize import crop_left_right, crop, pad
from misc.letterbox import LetterBox

def pre_transform_letterbox(im, sz_wh, **kwargs) -> List:
    # sz_wh = 640, 640
    args = dict(
        auto=False,
        scaleFill=False,
        scaleup=False,
        center=True
    )
    args.update(kwargs)
    letterbox = LetterBox(
        tuple(reversed(sz_wh)),
        **args
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
    
    im = np.stack(pre_transform_letterbox(im, sz_wh))
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


def _post_compute(fake_images):
    fake_images = torch.clamp(fake_images, -1., 1.)
    fake_images = (fake_images + 1) / 2.0  # [-1, 1] -> [0, 1]
    fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    fake_images = fake_images.astype(np.uint8)
    return fake_images