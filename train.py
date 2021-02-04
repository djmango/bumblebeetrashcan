import os
from pathlib import Path
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

HERE = Path(__file__).parent.absolute()

class ClockDataset(Dataset):
    """ Clock dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotations_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.annotations_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float')
        sample = {'image': image, 'annotations': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'annotations': annotations}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'annotations': torch.from_numpy(annotations)}

def clock():
    files = sorted(os.listdir(HERE.joinpath('traindata', 'clock')))
    dataset = ClockDataset(csv_file='traindata/clock/annotations.csv', root_dir='traindata/clock/')

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['image'].shape, sample['annotations'].shape)


if __name__ == '__main__':
    pass