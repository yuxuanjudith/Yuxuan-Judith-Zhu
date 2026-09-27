import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T


def get_dataloaders(
    dataset_name: str = "mnist",
    batch_size: int = 32,
    train_transforms=None,
    test_transforms=None,
):
    # usually, we want to split data into training, validation, and test sets
    # for simplicity, we will only use training and test sets
    if train_transforms is None:
        if dataset_name == "mnist":
            train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        elif dataset_name == "cifar100":
            train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
    if test_transforms is None:
        if dataset_name == "mnist":
            test_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        elif dataset_name == "cifar100":
            test_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    if dataset_name == "mnist":
        # Do Not Change This Code
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transforms
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transforms
        )

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "cifar100":
        # TODO: Load CIFAR100 dataset
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True,transform=train_transforms
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download= True, transform=test_transforms
        )
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return trainloader, testloader
