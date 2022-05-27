import torch 
import torch.nn as nn 
import torch.autograd as grad
import torch.nn.functional as F 
import numpy as np
import os.path as ospath
import os
import time 
import csv
from torch.utils.data import DataLoader 

from dataLoader import ModelNet40
from model import PointNetClassifier

def split_data():
    
    with open('ModelNet40.csv', 'r') as f:
        data = csv.reader(f, skipinitialspace=True)
        data_list = list(data)
        for row in data_list:
            curr = row[:2460]
            print(curr)
    
    #data = '/data/ModelNet40.csv'
    #return data
    
def main():

    num_points = 2000
    k = 3
    dataset_root_path = '/data/ModelNet40'
    model_path = 'classifier_model_path.pth'
    batch_size = 32

    # Then we instantiate dataset loader
    dataset = ModelNet40(dataset_root_path, test=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    gt_key = dataset.get_gt_key()


    # Instatiate networrk
    model = PointNetClassifier(num_points, k).eval().cuda().double()

    model.load_state_dict(torch.load(model_path))

    # To keep track of number of samples seen
    total_num_samples = 0
    class_num_samples = np.zeros(40)

    # We have 40 classes from ModelNet40 dataset
    # we create arrays of len = 40 to track per class accuracy
    class_correct = np.zeros(40)
    class_incorrect = np.zeros(40)


    # We also need to keep track of accuracy
    total_correct = 0
    total_incorrect = 0

    # To print feedback
    print("Starting evaluation... \n")
    print("Processing {} samples in batches of {}...".format(len(dataset), batch_size))

    num_batches = len(dataset)/batch_size

    for i, sample in enumerate(data_loader):
        print("Batch {} / {}".format(i, num_batches))

        # Then we parse through loaded data
        points = grad.Variable(sample[0]).cuda()
        target = grad.Variable(sample[1]).cuda()
        path = sample[2]

        # The forward pass
        #passing points into classifier model
        pred, _ = model(points)

        # Update accuracy
        # print predictions
        # print softmax of prediction. softmax(preds, dim=1)

        _, idx  = torch.max(F.softmax(pred, dim=1), 1)

        idx = idx.cpu().numpy()
        target = target.cpu().numpy()
        total_num_samples += len(target)
        for j in range(len(target)):
            val = target[j] == idx[j]
            total_correct +=val
            class_correct[target[j]] +=val
            total_incorrect +=np.logical_not(val)
            class_incorrect[target[j]] += np.logical_not(val)
            class_num_samples[target[j]] +=1

    
    print("Done!")
    print("Total Accuracy: {:2f}".format(total_correct/float(total_num_samples)))
    print("Per class accuracy: ")
    for i in range(len(class_correct)):
        print("{}: {:2f}".format(gt_key[i], class_correct[i]/float(class_num_samples[i])))




def test():
    print("No ERRS!")

if __name__ == "__main__":
    test()
    split_data()
    #main()