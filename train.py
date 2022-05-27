from random import shuffle
from sched import scheduler
import torch 
import torch.nn as nn
import numpy as np
import os.path as ospath 
import os 
import time 
from torch.utils.data import DataLoader
import torch.autograd as grad 
import torch.nn.functional as F 
import csv

from dataLoader import ModelNet40
from model import PointNetClassifier


def main():

    # Hyperparameters 
    num_points = 2000
    k = 3
    num_epochs = 60
    batch_size = 32
    lr = 0.001
    reg_weight = 0.002
    printout = 20
    dataset_root_path = 'data/ModelNet40/'
    snapshot = 10
    snapshot_dir = 'snapshots'


    try:
         os.mkdir(snapshot_dir)

    except:
         pass


    # Then we instantiate dataset loader
    train_data = ModelNet40(dataset_root_path)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    gt_key = train_data.get_gt_key()

    # Now we instantiate the network
    classifier = PointNetClassifier(num_points, k).train() # .cpu() #cuda().double()
    loss = nn.CrossEntropyLoss()
    regularization = nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )


    # Identity matrix for enforcing orthogonality of second transform
    identity = grad.Variable(torch.eye(64).double(),
		requires_grad=False) #cuda().cpu(), 

	# Some timers and a counter
    forward_time = 0.
    backprop_time = 0.
    network_time = 0.
    batch_counter = 0

	# Whether to save a snapshot
    save = False

    print ('Starting training...\n')

	# Run through all epochs
    for ep in range(num_epochs):

        if ep % snapshot == 0 and ep != 0:
            save = True

		# Update the optimizer according to the learning rate schedule
        scheduler.step()

        for i, sample in enumerate(data_loader):

			# Parse loaded data
            points = grad.Variable(sample[0])#.cpu()#cuda()
            target = grad.Variable(sample[1])#.cpu()#cuda()

			# Record starting time
            start_time = time.time()

			# Zero out the gradients
            optimizer.zero_grad()

			# Forward pass
            pred, T2 = classifier(points)

			# Compute forward pass time
            forward_finish = time.time()
            forward_time += forward_finish - start_time

			# Compute cross entropy loss
            pred_error = loss(pred, target)

			# Also enforce orthogonality in the embedded transform
            reg_error = regularization(
				torch.bmm(T2, T2.permute(0,2,1)), 
				identity.expand(T2.shape[0], -1, -1))

			# Total error is the weighted sum of the prediction error and the 
			# regularization error
            total_error = pred_error + reg_weight * reg_error

			# Backpropagate
            total_error.backward()

			# Update the weights
            optimizer.step()

			# Compute backprop time
            backprop_finish = time.time()
            backprop_time += backprop_finish - forward_finish

			# Compute network time
            network_finish = time.time()
            network_time += network_finish - start_time

			# Increment batch counter
            batch_counter += 1

			#------------------------------------------------------------------
			# Print feedback
			#------------------------------------------------------------------

            if (i+1) % printout == 0:
				# Print progress
                print ('Epoch {}/{}'.format(ep+1, num_epochs))
                print ('Batches {}-{}/{} (BS = {})'.format(i-printout+1, i,
					len(train_data) / batch_size, batch_size))
                print ('PointClouds Seen: {}'.format(
					ep * len(train_data) + (i+1) * batch_size))
				
				# Print network speed
                print ('{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop'))
                print ('  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
					.format(network_time, forward_time, backprop_time))

				# Print current error
                print ('{:16}[ {:12}{:12} ]'.format('Total Error', 
					'Pred Error', 'Reg Error'))
                print ('  {:<14.4f}[ {:<10.4f}  {:<10.4f} ]'.format(
					total_error.data[0], pred_error.data[0], reg_error.data[0]))
                print ('\n')

				# Reset timers
                forward_time = 0.
                backprop_time = 0.
                network_time = 0.

            if save:
                print ('Saving model snapshot...')
                save_model(classifier, snapshot_dir, ep)
                save = False
                

def save_model(model, snapshot_dir, ep):
	save_path = ospath.join(snapshot_dir, 'snapshot{}.params' \
		.format(ep))
	torch.save(model.state_dict(), save_path)	

def test():
    print("Runtime ERRs test!")

if __name__ == "__main__":
    test()
    main()

