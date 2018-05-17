Dataset used - BreakHis 

The BreakHis Dataset is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). It contains 2,480  benign and 5,429 malignant samples . 

The dataset BreaKHis is divided into two main groups: Benign tumors and Malignant tumors .  A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body. A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body. Hence benign tumors on rare occasion may actually be life-threatening, and general rule aren't as bad as the malignant tumors. 

The images are divided on the basis of patients . The images of approximately 75% patients (benign and malignant combined) are included in the training data and testing data comprises of rest . 

Tensorflow is used in the backend in Convolutional Neural Network . A testing accuracy of more than 80% is achieved with 5 epochs.

'Occlusion' is used to indicate the tissue in the image which is responsible for the cancer . A random image from testing set is selected. A black window is passed all over the image and the image passed in the CNN for each position of the black window . Based on the performance of neural network , we come to know what the model looks in that image to classify it . This is the most important part i.e tissue in that image which is responsible most for cancer . A cmap is used to plot these parts .  
