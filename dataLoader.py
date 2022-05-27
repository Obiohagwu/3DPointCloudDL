
import torch 
import torch.utils.data 
import numpy as np
import os 
import os.path as ospath 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 




# Creating a class to extract modelnet 40 dataset from an os path
class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path, test=False):
        self.test = test 

        # Build path list
        self.input_pairs, self.gt_key = self.create_input_list(
            dataset_root_path, test
        )
        
    def __len__(self):
        return len(self.input_pairs)

    def __getitem__(self, idx):
        # We select a path
        path, label = self.input_pairs[idx]

        # Then we parse the vertices of he file
        vertices = self.off_vertex_parser(path)
        if not self.test:
            vertices = self.augment_data(vertices)

        # then we convert from numpy format to torch variable
        return[torch.from_numpy(vertices), label, path]

    
    def get_gt_key(self):
        return self.gt_key

    
    def create_input_list(self, dataset_root_path, test):
        input_pairs = []

        # List of tuples grouping a label with a class
        gt_key = os.listdir(dataset_root_path)

        for idx, obj in enumerate(gt_key):
            # Bool if for if test set or train set
            if test:
                path_to_files = ospath.join(dataset_root_path, obj, 'test')
            else:
                path_to_files = ospath.join(dataset_root_path, obj, 'train')


            files = os.listdir(path_to_files)
            filepaths = [(ospath.join(path_to_files, file), idx) for file in files]

            input_pairs = input_pairs + filepaths

        return input_pairs, gt_key


    def augment_data(self, vertices):
    # Apply a random rotation about y-axis
        theta = 2 * np.pi * np.random.rand(1)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]], dtype=object)

        vertices = np.matmul(Ry, vertices)

        # Add some gaussian noise with sd=0.02
        vertices += np.random.normal(scale=0.02, size=vertices.shape)

        return vertices

    def off_vertex_parser(self, path_to_off_file):
        # Read the Off file
        with open(path_to_off_file, 'r') as f:
            contents = f.readlines()

        # Find the number of vertices contained
        if contents[0].strip().lower() !='off':
            num_vertices = int(contents[0].strip()[4:].split(' ')[0])
            start_line = 1
        else: 
            num_vertices = int(contents[1].strip().split(' ')[0])
            start_line = 2
        
        # Then we convert all the vertex lines to a list of lists --> matrix
        vertex_list = [map(float, contents[i].strip().split(' ')) for i in range(start_line, start_line+num_vertices)]

        # retuen the vertices as a 3 x N np arr
        return np.array(vertex_list)


def test():
    print("To test runtime errs!")

if __name__ == '__main__':
    test()