
# Overview

This repository include the binary executables and Python codes for reproducing the results of the paper
"Predicting Defects in Laser Powder Bed Fusion using *in-situ* Thermal Imaging Data and Machine Learning", 
[found here](https://www.sciencedirect.com/science/article/pii/S2214860422004018), published in the ***Journal of Additive Manufacturing***.


## Data Processing

The core part of this work is the data post-processing procedure. After the data registration, where 
each voxel with the coordinates `X, Y, Z` and thermal features including &tau; and T is mapped into its binary label as
0 or 1 (healthy or defective), the unfolding process is performed on the dataset. Unfolding means 
using sliding kernels ***K3, K5, K7*** to include the thermal features of each voxel itself and features 
of 1st, 2nd and 3rd nearest neighbor voxels around it. Then, Voxels near the top, bottom and side surfaces 
of the built volume are excluded from the data if they do not have the complete set of neighbors (this is because 
the voxels near the surface and boundary do not participate in the heat transfer during LPBF or they have different
heat transfer physics comparing to the voxels far from the boundaries. The following figure show the processes of 
data post-processing, data unfolding and then training, validation, testing path: 

![The following figure show the processes of data post-processing, data unfolding and then 
training, validation, testing path:](https://github.com/sinaDFT/ML-LPBF-AM/blob/1989cb5f1559f6fb86ffd86978b8750f040f8b90/Process.PNG)

### ***Four*** main steps for data post-processing

There are four different steps for the post processing of the raw data to result in clean and proper dataset 
to be fed into the ML models which are as follow:

I. Indexing the coordinates, features and labels of the points and creating a 3D image where points are converted to grid image pixels.

II. Assigning -1 to the pixels inside the square gride where there is no labeled pixel data and it covers around
the boundary pixels. This is a small part of an image as most part of the image includes the pixels with binary values, `[0, 1]`.

III. 	Unfolding the labels, features and coordinates in x,y and z directions with different windowing kernels based on the idea 
of convolutional kernels to take into account the features of the different nearest neighbors around a central point. Through this process, 
the 3D image will be patched using various kernel sizes. We use kernels with dimension 
$k_1 \times k_1 \times k_2,{(k_1=3, [\forall k_2|k_2 \in (1,3)]),(k_1=5, [\forall k_2|k_2 \in (1,3,5)]),(k_1=7, [\forall k_2|k_2 \in (1,3,5,7)])}$.

IV. After unfolding the data with desired kernel, we need to remove pixels with -1 values in the patched data after unfolding. 
In the last step, indexes corresponding to -1 values are recognized and the framing kernel including the -1 is removed.

### Explaining each step of data post-processing

#### I. *Indexing*

Post processing the data is the most important part of any ML method to provide the model with proper and clean dataset. In this work, 
the available data is `N` points with their 3D coordinates, two features for each point and their labels, `0` or `1`, as fully dense or defective. 
Our goal is to use different nearest neighbor features to predict the label of each target pixel based on the physics of the 
manufacturing process and provide the model with more information to help it to have better predictions. So, the first step is converting 
the points to pixels on a 3D rectangular cube grid (square grid in 2D). It could be a 3D cube if the number of dimensions in `x, y, z` directions 
are equal. To do this, it is needed to index coordinates, features and labels of all data points through the process of indexing. 
Indexing the labels of points based on their 3D cartesian coordinates will map them to the grid pixels and it provides images with 
dimension $26 \times 26$ on top of each other. Considering the 3d cylinder like sample, after indexing, it is a 3D image with dimension of
$26 \times 26 \times 398$. In this process, two functions are defined to calculate the length of the data, in different directions and 
the conversion of coordinates to indexes as $F(\Delta x, \Delta y, \Delta z, x, y, z)$ and $G(\Delta x, \Delta y, \Delta z, x, y, z)$,
respectively, using the $\Delta x, \Delta y, \Delta z$ and the coordinates of each point in 3D Euclidean space as the variables of functions. 
So, the length of data would be a `numpy` array with dimension $(m,n,k)$.

#### I. *Assigning the data values to indexes*

 



   

        


    
