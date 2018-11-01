from __future__ import print_function

import torch.utils.data as data
import os
import random
import glob
from PIL import Image
from utils import preprocess

_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}


class Cityscapes(data.Dataset):
  CLASSES = [
      'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
      'truck', 'bus', 'train', 'motorcycle', 'bicycle'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size

    if download:
      self.download()

    dataset_split = 'train' if self.train else 'val'
    self.images = self._get_files('image', dataset_split)
    self.masks = self._get_files('label', dataset_split)

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])

    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size) if self.train else (1025, 2049))

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = self.target_transform(_target)

    return _img, _target

  def _get_files(self, data, dataset_split):
    pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
    search_files = os.path.join(
        self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')
