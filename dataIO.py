import torch
from torchvision import datasets, transforms

def get_mnist_img(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/', train=True, download=True,
                       transform =transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5,), std=(0.5,))
                       ])),
        batch_size=batch_size,
        shuffle=True)

    images, labels = next(iter(train_loader))
    return images

def get_z(batch_size, z_size):
    z = torch.normal(0, 1, size=(batch_size, z_size))
    return z










