import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
import numpy as np

# Transformer part of the pointNet architecture is a critical component

# Transfomer class used to compute a KxK affine transform from the input data
# Used to transform inputs to a "Canonical view" --  Meaning

#

class Transformer(nn.Module):
    def __init__(self, num_points=2000, K=3):
        super(Transformer, self).__init__()
        
        # We take Nxk=3 input


        # K = 3 is (x,y,z) and our network should be permutationlly invariant to diff combinations
        self.K = K

        # Size of input 
        # N inputs from architecture
        self.N = num_points

        # Initialize identify matrix on gpu
        self.identify = grad.Variable(
            torch.eye(self.K).double().view(-1)#.cuda()
        )

        # first block for transformation to get Nx64
        self.block1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # second block transform from 64 to 128
        self.block2 = nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # third block transform from 128 to 1024
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Then the final MLP layer that outputs (B=512, 1024, K=3)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, K*K)
        )

    # We are takin B x N x K input. B batches, N points and K=3 dims(x,y,z)

    def forward(self, x):

        # Compute feature extraction
        # Output will be B x 1024 x N

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # We apply the max pool to the transformed points
        # Output --> B x 1024 x 1 --> B x 1024(after squeeze)
        x = F.max_pool1d(x, self.N).squeeze(2)

        # Then we send the maxpooled features through mlp layer
        # Output = B * k^2
        x = self.mlp(x)

        # Add identity matrix to transform
        # Output is B x K^2 (broadcasting takes care of batch dimension)
        x+=self.identify

        # Then we reshape output into B x K x K affine transformation matrices

        x = x.view(-1, self.K, self.K)

        return x



def test():
    print('Just testing for any runtime errors!')


if __name__ == "__main__":
    test()