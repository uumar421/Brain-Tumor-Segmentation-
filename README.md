# Brain Tumor Segmentation 
Implementation of the segmentation model created as part of Final Year Project titled as "Brain Tumor Segmentation using Encoder Decoder based Fully Convolutional Network".

# Summary
The implemented method is an encoder decoder-based fully convolutional network that efficiently predicts and segments the tumors from 3D MRI scans. Inside the network, residual extended skips (Resnet blocks) are incorporated through addition of identity skip connections in order to minimize the issue of vanishing gradient during backpropagation that most neural networks face in case of deep image processing. Apart from that, comprehensive augmentation and pre-processing is applied to ensure that our network is able to process and learn well from the 3D training images.  For training and evaluation, BraTS 2020 training dataset is used by splitting it into a training and validation set. The method is implemented in Python 3 using the PyTorch framework.   Three tumor sub-regions to be segmented are whole tumor, tumor core and enhancing tumor core. The results of our method show high segmentation accuracy and efficiency in predictions of brain tumors with dice scores of 81.8%, 82.4% and 84.7% for whole tumor, tumor core and enhancing tumor core respectively. 
 
# Libraries Required:
  -> PyTorch 
  -> NumPy 
  -> NiBabel 
  -> Matplotlib
  -> OpenCV
  -> Scikit-image
  
# Architecture
The architecture implemented in this project, originally proposed by Myronenko et al. in the article titled as "Robust Semantic Segmentation of Brain Tumor
Regions from 3D MRIs", is illustrated with the help of following figure:

![image](https://user-images.githubusercontent.com/69485235/128057854-432de3c2-951e-4dba-a7c8-bd6b2748126c.png)
