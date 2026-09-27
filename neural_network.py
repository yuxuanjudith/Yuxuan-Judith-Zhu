import torch
from torch import nn


class FCModel(nn.Module):
    """
    Part 1.a: Fully connected neural network
    """

    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10):
        # MNIST and CIFAR100 have different image size/number of classes,
        # so you may want to pass in additional arguments
        # to allow the same model code adapt to different datasets
        super(FCModel, self).__init__()
        # TODO: Define the layers for the fully connected neural network
        # Use nn.Flatten, nn.Linear, and nn.ReLU appropriately
        # expected architecture:
        # flatten -> Linear(c=256) -> ReLU -> Linear(c=256) -> ReLU -> Linear
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the fully connected neural network
        return self.net(x)


class CNNModel(nn.Module):
    """
    Part 1.b: Convolutional neural network
    """

    def __init__(self, in_channels : int = 1, img_size: int = 28, num_classes : int = 10):
        # MNIST and CIFAR100 have different image size/number of classes,
        # so you may want to pass in additional arguments
        # to allow the same model code adapt to different datasets
        super(CNNModel, self).__init__()
        # TODO: Define the convolutional and fully connected layers for the CNN
        # Use nn.Conv2d, nn.MaxPool2d, nn.Linear, Flatten, and nn.ReLU appropriately
        # expected architecture (c=channels, s=stride):
        # Conv(c=32, s=1) -> ReLU -> MaxPool(s=2) -> Conv(c=64, s=1) -> ReLU -> flatten -> Linear
        # You may tune unspecified hyperparameters like kernel size, initialization, etc.
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride= 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride=1, padding = 1)
        self.relu2 = nn.ReLU()
        self.flatten= nn.Flatten()
        pooled_size = img_size//2
        self.linear= nn.Linear(64*pooled_size*pooled_size, num_classes)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x



class ClassificationLoss(nn.Module):
    """
    Part 2: Loss function

    Implement softmax cross entropy loss without using torch.nn
    """

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        # TODO: Define the loss function for classification

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the loss function
        # y_pred: (N, num_classes) raw logits (NOT softmaxed)
        # y_true: (N,) integer class labels
 
        # log-sum-exp over the class dimension, computed in a numerically
        # stable way (subtract the max logit before exponentiating)
        max_logits, _ = torch.max(y_pred, dim=1, keepdim=True)
        stable_logits = y_pred - max_logits
        log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=1, keepdim=True)) + max_logits
 
        # log-probability of the correct class for each sample
        correct_class_logits = y_pred[torch.arange(y_pred.size(0)), y_true].unsqueeze(1)
        log_probs = correct_class_logits - log_sum_exp
 
        # negative log likelihood, averaged over the batch
        loss = -log_probs.mean()
        return loss


class BasicResidualBlock(nn.Module):
    """
    A standard ResNet "BasicBlock" (used in ResNet18/34): two 3x3 convs with a
    skip connection. If the spatial size or channel count changes, the skip
    connection is projected with a 1x1 conv so the shapes match for addition.
    """
 
    expansion = 1
 
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class BetterModel(nn.Module):
    """
    Part 3: Better Model: ResNet
    """

    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super(BetterModel, self).__init__()
        self.in_planes = 64
 
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
 
        # ResNet-18 layout: 2 blocks per stage, channels 64 -> 128 -> 256 -> 512
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicResidualBlock.expansion, num_classes)
 
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicResidualBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicResidualBlock.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the custom model
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
