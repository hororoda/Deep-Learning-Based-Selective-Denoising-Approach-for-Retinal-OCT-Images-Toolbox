# Deep-Learning-Based-Selective-Denoising-Approach-for-Retinal-OCT-Images-Toolbox
This toolkit includes three parts: (1) noise_level.py is used as a criterion for dataset segmentation. (2) When the datasets required to train the model are ready, you can train the model using train.py. The trained model can be loaded into the screen.py to screen NRR and NF images. (3) In addition, several evaluation metrics are provided to comprehensively assess the model’s performance.

# Guide:
CUDA Toolkit and Pytorch are needed, please make sure the environment is set up before running. Some files require specific version of torchvision to run. For specific version information, please refer to the note. Retinal OCT image datasets can be downloaded from the source listed in the data section. 

# Python files:
## train.py
Main function for training models to classify retinal OCT images into NRR and NF images, and recording accuracy and loss for every epoch.
## screen.py
Main function for classifying retinal OCT images into NRR and NF images using the model trained using train.py.
## MyImageFolder.py
A custom class inherits from class ‘ImageFolder’ to read files after filtering.
## confusion_matrix.py
A function for calculating and plotting confusion matrix of the model.
## plot_accu_curve.py
A function for plotting accuracy curves of the model. (.csv files needed can be downloaded from tensorboard.)
## calculate_evaluation_metrics.py
A function for calculating Recall, Specificity, Precision, F1-score, Overall Accuracy, Macro and Micro F1-scores.
## plot_roc_curve.py
A function for plotting ROC curves and calculating AUC values.
## grad_cam.py
A function for plotting Grad_CAMs of the model.
## noise_level.py
Liu X, Tanaka M, Okutomi M. Single-image noise level estimation for blind denoising[J]. IEEE transactions on image processing, 2013, 22(12): 5226-5237.

# Data:
## Dataset.1
Kermany D S, Goldbaum M, Cai W, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning[J]. Cell, 2018, 172(5): 1122-1131. e9.
## Dataset.2
Gholami P, Roy P, Parthasarathy M K, et al. OCTID: Optical coherence tomography image database[J]. Computers & Electrical Engineering, 2020, 81: 106532.
## Dataset.3
https://www.kaggle.com/datasets/hadeersaif/retinal-oct5classes
