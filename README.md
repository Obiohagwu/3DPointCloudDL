# 3DPointCloudDL
A mini-project applying DL techniques to 3D point cloud data for classification and segmentation

## Background
An implementation of PointNet architecture for 3D point cloud classificatiion and segmentation.
Methods that preserve permutation invariance of cloud high-dimensional cloud data points. 

The crux of the problem solved by PointNet architecture is learning 3D representation of unordered sets of point data in (x,y,z) Euclidean space. Unlike other deep learning approeaches that deal with tokenized sequences (for NLP tasks) and/or image pixel raster values (for image tasks), permutation invariance and euclidean symmetry is important for learning high-dimensional representations of unorderd sets of coordinate data.

### Some properties of Point Sets in n-dimensional space of R:

 - Given that they are unordered, any such network would have to implement some sort of symmetry function that would such that N input datapoints would be invariant to N! permutation of itself.
 - Given tht point clouds hold spatial structure and local relationships given projection onto euclidean space, any suc model would have to be able to learn representatin preesrving spatial structure between local/neighbouring points.
 - Given that we are applying thees models unto geometric structures, our model should account for invarance of projection under geometric transformation while still preserving spatial structure.

## Main Components of models

The model used to implement PointNet are slpit into 3 component parts:

- Baseline Transform
- Base pointNet
- PointNet model

### Baseline Transform

We will need to implement a Transform class that inherits from pytorch nn.Module. 


### Base PointNet





### PointNet Model






## Original Paper

Citation for original paper this re-implementation is based on:






> Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." 
> Proc. Computer Vision and Pattern Recognition (CVPR), IEEE 1.2 (2017): 4.


