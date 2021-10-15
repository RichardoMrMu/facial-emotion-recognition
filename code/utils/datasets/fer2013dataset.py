import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from utils.augmenters.augment import seg
import torch
from utils.autoaugment import CIFAR10Policy,ImageNetPolicy,Cutout
EMOTION_DICT = {
    0: "angry", # 0.2513149186666202
    1: "disgusted", # 0.01518687519593159
    2: "fearful", # 0.14270786164617366
    3: "happy", # 0.2513149186666202
    4: "sad", # 0.16823992476226968
    5: "surprise", # 0.11045316799609878
    6: "neutral", # 0.17294228290779895
}


class FER2013(Dataset):
    """
    args:
        k: k-folder cross validation
    """
    def __init__(self, stage, configs, tta=False, tta_size=48,k=0):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])

        self._data = pd.read_csv(os.path.join(configs["data_path"], "{}.csv".format(stage)))

        self._pixels = self._data["pixels"].tolist()
        self._emotions = pd.get_dummies(self._data["emotion"])

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self._image_size),
                transforms.ToTensor(),
            ]
        )
        self._transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self._image_size),
            transforms.ToTensor(),
        ])
        self._temp_train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self._image_size),
                transforms.RandomCrop(self._image_size[0], padding=16),
                transforms.RandomHorizontalFlip( 0.5),
                CIFAR10Policy(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=64),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        self._temp_test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self._image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):
        pixels = self._pixels[idx]
        pixels = list(map(int, pixels.split(" ")))
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)

        # image = cv2.resize(image, self._image_size)
        image = np.dstack([image] * 3)
        if self._stage == "train":
            image = seg(image=image)
            # pass
        if self._stage == "test" and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            if self._configs['new_transform']:
                images = list(map(self._temp_test_transform, images))
            else:
                images = list(map(self._transform_test, images))
            target = self._emotions.iloc[idx].idxmax()
            return images, target
        if self._configs['new_transform']:
            image = self._temp_train_transform(image)
        else:
            image = self._transform(image)
        target = self._emotions.iloc[idx].idxmax()
        return image, target


def fer2013(stage, configs=None, tta=False, tta_size=48):
    return FER2013(stage, configs, tta, tta_size)


if __name__ == "__main__":
    data = FER2013(
        "train",
        {
            "data_path": "/home/z/research/tee/saved/data/fer2013/",
            "image_size": 224,
            "in_channels": 3,
        },
    )
    import cv2
    from barez import pp

    targets = []

    for i in range(len(data)):
        image, target = data[i]
        cv2.imwrite("debug/{}.png".format(i), image)
        if i == 200:
            break
