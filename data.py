import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T
import torch.nn.functional as F

import config as c
import torchvision.datasets
import imagenet as img


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if c.add_image_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)

class RandomHorizontalFlipTensor(object):
    """Random horizontal flip of a CHW image tensor."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        assert img.dim() == 3
        if np.random.rand() < self.p:
            return img.flip(2) # Flip the width dimension, assuming img shape is CHW.
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


if c.dataset == 'cifar-10':

    data_dir = './data/cifar-10_data/'

    means = torch.tensor([0.4914, 0.4822, 0.4465]).view([1, 3, 1, 1]).cuda()
    stds  = torch.tensor([0.2023, 0.1994, 0.2010]).view([1, 3, 1, 1]).cuda()

    transform_train = T.Compose([T.Resize(c.img_dims[1]), T.RandomHorizontalFlip(), T.ToTensor(), add_noise])
    transform_test  = T.Compose([T.Resize(c.img_dims[1]), T.ToTensor()])

    train_data = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform_train, download=True)
    test_data  = torchvision.datasets.CIFAR10(data_dir, train=False, transform=transform_test, download=True)

    train_loader  = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=c.workers,
                               pin_memory=True, drop_last=True)
    test_loader   = DataLoader(test_data,  batch_size=c.batch_size, shuffle=False, num_workers=c.workers,
                               pin_memory=True, drop_last=False)

elif c.dataset == 'ImageNet':

    if c.mode == 'pre_training':

        root            = './data/imagenet64/'
        dataset_class   = img.ImageNet64
        train_transform = T.Compose([T.ToTensor(),
                                     RandomHorizontalFlipTensor(),
                                     ReshapeTransform([c.img_dims[0], c.img_dims[1], c.img_dims[2]])])
        train_data      = dataset_class(root=root, train=True, download=False, transform=train_transform)
        train_loader    = DataLoader(train_data, batch_size=c.batch_size, shuffle=False, num_workers=c.workers,
                                  pin_memory=True,
                                  drop_last=True)

    else:
        data_dir = './data/ImageNet_data/'
        means    = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).cuda()
        stds     = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).cuda()

        val_data   = torch.load(data_dir + 'imagenet.pth')
        val_data   = F.interpolate(val_data, size=c.org_size)
        val_labels = torch.load(data_dir + 'imagenet_labels.pth').to(torch.long)

        test_data   = TensorDataset(val_data, val_labels)
        test_loader = DataLoader(test_data,  batch_size=c.batch_size, shuffle=False, num_workers=c.workers,
                                 pin_memory=True, drop_last=False)

elif c.dataset == 'svhn':

    data_dir = './data/svhn_data/'

    means = 0.5
    stds  = 0.5

    train_data = torchvision.datasets.SVHN(data_dir, split='train', transform=T.ToTensor(), download=True)
    test_data  = torchvision.datasets.SVHN(data_dir, split='test', transform=T.ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=c.workers,
                              pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_data,  batch_size=c.batch_size, shuffle=False, num_workers=c.workers,
                              pin_memory=True, drop_last=True)

elif c.dataset == 'CelebA':

    data_dir = './data/celeba_data/'

    means = 0.5
    stds  = 0.5

    trans = T.Compose([CropTransform((25, 50, 25 + 128, 50 + 128)), T.Resize(c.img_dims[1]), T.ToTensor(),
                       ReshapeTransform([c.img_dims[0], c.img_dims[1], c.img_dims[2]])])

    train_data = torchvision.datasets.CelebA(data_dir, split='train', transform=trans, download=True)
    test_data  = torchvision.datasets.CelebA(data_dir, split='test',  transform=trans, download=True)

    train_loader  = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=c.workers,
                               pin_memory=True, drop_last=True)
    test_loader   = DataLoader(test_data,  batch_size=c.batch_size, shuffle=False, num_workers=c.workers,
                               pin_memory=True, drop_last=True)

else:
    raise ValueError('Dataset {} is not defined!'.format(c.dataset))

