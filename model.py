import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
import numpy as np
from basePointNet import BasePointNet


# Now we are going to implement the pointnet classifier using components
# defined in basepointnet for global feature and local embedding extraction  &
# transformsScript for permutationally invariant trasformation of k=x,y,z point coords


class PointNetClassifier(nn.Module):
    def __init__(self, num_point=2000, K=3):
        super(PointNetClassifier, self).__init__()

        # initialize local and global feature extractor
        self.basePoint = BasePointNet(num_point, K)


        # Then we initalize classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 40)
        )

    

    # Takes B x K x N as input. B batches K=3 (x,y,z) dims and N points
    def forward(self, x):

        x, _, T2 = self.basePoint(x)

        # return B x 40
        return self.classifier(x), T2


def test():
    print("just cheacking for runtime err!")


if __name__ == "__main__":
    test()