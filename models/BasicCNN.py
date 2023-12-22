# THIS IS THE BASIC, GIVEN MODEL

import torch.nn as nn
import torch.nn.functional as F


# Define the CNN architecture
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # pooling has no learnable parameters, so we can just use one
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP classifier
        self.fc = nn.Linear(128 * 3 * 3, num_classes)

    def forward(self, x):
        # print("Input size:", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv5(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv6(x)))
        # print("Layer size:", x.size())
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor
        # print("Layer size:", x.size())

        # Fully connected layer for classification
        x = self.fc(x)

        return x
