
# Overview

This repository include the binary executables and Python codes for reproducing the results of the paper
"Predicting Defects in Laser Powder Bed Fusion using *in-situ* Thermal Imaging Data and Machine Learning", 
[found here](https://www.sciencedirect.com/science/article/pii/S2214860422004018), published in the ***Journal of Additive Manufacturing***.


## Data Processing

The core part of this work is the data post-processing procedure. After the data registration, where 
each voxel with the coordinates `X, Y, Z` and features `tau, T` is mapped into its binary label as
0 or 1 (healthy or defective), the unfolding process is performed on the dataset. Unfolding means 
using sliding kernels `***K3, K5, K7***` to include the thermal features of each voxel itself and features 
of 1st, 2nd and 3rd nearest neighbor voxels around it. Then, Voxels near the top, bottom and side surfaces 
of the built volume are excluded from the data if they do not have the complete set of neighbors (this is because 
the voxels near the surface and boundary do not participate in the heat transfer during LPBF or they have different
heat transfer physics comparing to the voxels far from the boundaries. 

![The following figure show the processes of data post-processing, data unfolding and then 
training, validation, testing path:](https://github.com/sinaDFT/ML-LPBF-AM/blob/1989cb5f1559f6fb86ffd86978b8750f040f8b90/Process.PNG)

        


    
