# Brain Tumor Segmentation 

# Summary
The implemented method is an encoder decoder-based fully convolutional network that efficiently predicts and segments the tumors from 3D MRI scans. Inside the network, residual extended skips (Resnet blocks) are incorporated through addition of identity skip connections in order to minimize the issue of vanishing gradient during backpropagation that most neural networks face in case of deep image processing. Apart from that, comprehensive augmentation and pre-processing is applied to ensure that our network is able to process and learn well from the 3D training images.  For training and evaluation, BraTS 2020 training dataset is used by splitting it into a training and validation set. The method is implemented in Python 3 using the PyTorch framework.   Three tumor sub-regions to be segmented are whole tumor, tumor core and enhancing tumor core. The results of this method show high segmentation accuracy and efficiency in predictions of brain tumors with dice scores of 81.8%, 82.4% and 84.7% for whole tumor, tumor core and enhancing tumor core respectively. 
 
# Libraries Required:
  -> PyTorch 
  -> NumPy 
  -> NiBabel 
  -> Matplotlib
  -> OpenCV
  -> Scikit-image
  -> Monai
  
# Architecture
The architecture implemented in this project, originally proposed by Myronenko et al. in the article titled as "Robust Semantic Segmentation of Brain Tumor
Regions from 3D MRIs", is illustrated with the help of following figure:

![image](https://user-images.githubusercontent.com/69485235/128057854-432de3c2-951e-4dba-a7c8-bd6b2748126c.png)

Note: Every input is 5-dimensional. The batch dimension is not shown in the above figure.

# Results
The model is trained on Nvidia server platform containing four Nvidia Tesla V100 (32 GBs) GPUs for 300 epochs and the highest average dice score (accuracy) of 0.818 (81.8%) is achieved. 

![image](https://user-images.githubusercontent.com/69485235/128058395-c1152138-71a8-48cd-af00-63e15ef71323.png)

The following figure shows the visualization of the results achieved through the model.

![image](https://user-images.githubusercontent.com/69485235/128058532-eb548305-2c3b-49f5-8894-d4db4c80622d.png)


