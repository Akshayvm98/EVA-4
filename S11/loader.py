import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensor


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


class CustomAlbumentation():

    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(
                mean=mean,
                std=std,
                always_apply=True
            ),
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomCrop(height=32, width=32,p=0.5),
            A.Flip(p=0.5),
            A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, p=0.5),
            ToTensor(),
        ])

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image



'''train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)'''



test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CustomAlbumentation())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

dataloader_args_for_plot = dict(shuffle=True, batch_size=4, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
trainloader_for_plot = torch.utils.data.DataLoader(trainset, **dataloader_args_for_plot)

# test dataloader
testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
testloader_for_plot = torch.utils.data.DataLoader(testset, **dataloader_args_for_plot)
