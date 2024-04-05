import torch
import numpy as np
import random
from torch.utils.data import Dataset

from scipy import ndimage


# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


## Temporary
class ISICDataset(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train=True, Test=False):
        super(ISICDataset, self)
        self.train = train
        if train:
            self.data = np.load(path_Data + 'data_train.npy')
            self.mask = np.load(path_Data + 'mask_train.npy')
        else:
            if Test:
                self.data = np.load(path_Data + 'data_test.npy')
                self.mask = np.load(path_Data + 'mask_test.npy')
            else:
                self.data = np.load(path_Data + 'data_val.npy')
                self.mask = np.load(path_Data + 'mask_val.npy')

        self.data = dataset_normalized(self.data)
        self.mask = np.expand_dims(self.mask, axis=3)
        self.mask = self.mask / 255.

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)

        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute(2, 0, 1)
        seg = seg.permute(2, 0, 1)

        return img.float(), seg.float()

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label

    def __len__(self):
        return len(self.data)



def build_dataset(args):
    train_dataset = ISICDataset(path_Data=args.data_root, train=True)
    valid_dataset = ISICDataset(path_Data=args.data_root, train=False)
    return train_dataset, valid_dataset