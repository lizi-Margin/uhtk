import torch, cv2
import numpy as np
from typing import Union, List, Tuple
from torchvision.transforms import Compose
import torchvision.transforms as transforms

def debug(x):
    print("X")
    return x

class PrepareForNet(object):
    def __call__(self, sample):
        image = np.transpose(sample, (2, 0, 1))
        sample = np.ascontiguousarray(image).astype(np.float32)
        return sample


center_transform_test = Compose(
    [
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample).cuda(),
        lambda img: (img / 255.0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

center_transform_train = Compose(
    [
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample).cuda(),
        lambda img: (img / 255.0),
        # transforms.RandomRotation(degrees=(-0.5, -0.5), fill=(0, 0, 0)),
        transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
        transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
        transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

_showed = False
def preprocess(imgs: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
    assert isinstance(imgs, list)
    assert isinstance(imgs[0], (tuple, list,))

    # center_imgs = np.array([cls.get_center_(frame.copy()) for frame in imgs])
    # map_imgs = np.array([cls.get_map(frame.copy()) for frame in imgs])

    def trans(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = center_transform_train if train else center_transform_test
        image = transform(image)
        assert image.shape[0] == 3
        return image
    imgs_t = torch.stack([trans(img) for img in imgs])

    

    global _showed
    if not _showed and train:
        for i in range(5):
            center_0 = ((imgs_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
            cv2.imshow('raw', center_0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        _showed = True
    return imgs_t

