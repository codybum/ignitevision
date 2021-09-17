import os

from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

train_transform = Compose(
    [
        Resize(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = Compose(
    [
     Resize(224),
     ToTensor(),
     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_train_test_datasets_old(path):
    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    return train_ds, test_ds

def get_train_test_datasets(path):

    train_ds = datasets.ImageFolder(os.path.join(path, 'train'),train_transform)
    test_ds = datasets.ImageFolder(os.path.join(path, 'val'), train_transform)

    return train_ds, test_ds

def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    return fn(num_classes=10)
