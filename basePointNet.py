from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad 

from transformsScript import Transformer


"""

The use of this base pointnet class is to extract local emedding structures and 
global features from input point sets

"""

class BasePointNet(nn.Module):
    def __init__(self, num_points=2000, K=3):
        super(BasePointNet, self).__init__()

        # Recall the transformer we defined in transformersScript.py
        # We apply it here as seen in architecture
        # Input transformer for k-dim input (x,y,z)
        self.input_transformer = Transformer(num_points, K)

        # Embedding transformer is always usually 64 dimeantal
        self.embedding_transformer = Transformer(num_points, 64)


        # We implement 1d convolutional mlps. We are mapping k inputs to 64 outputs
        # so we can consider each 64 k-dim filter as defining a weight matrix for each point dim(x,y,z)
        # to each index of 64 dim embedding

        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()

        )




    def forward(self, x):
        
        # we get number of point N
        N = self.shape[2]


        # First we compute input data transform and transform the data 
        # T1 is B x K x K and x is B x K x N, so output after matrix multiplicatin is B x K x N

        T1 = self.input_transformer(x)
        x = torch.bmm(T1, x)

        # Run the transformed input through firt embedding mlp
        # Output is B x 64 x N
        x = self.mpl1(x)

        # Transform the embedding. This gives us "local embeddings"
        # T2 is B x 64 x 64 and x is B x 64 x N, so output is B x 64 x N
        T2 = self.embedding_transformer(x)
        local_embedding = torch.bmm(T2, x)

        # Then we embed local embedding. We extract global features from 
        # local embedding after mlp2 is applied
        # Output should be B x 1024 x N
        global_features = self.mlp2(local_embedding)



        # We pool over the number of poitns. This results in "global features"
        # Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
        global_features = F.max_pool1d(global_features, N).squeeze(2)

        return global_features, local_embedding, T2



def test():
    print("just cheacking for runtime err!")


if __name__ == "__main__":
    test()