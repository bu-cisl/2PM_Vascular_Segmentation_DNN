# 2PM Vascular Segmentation DNN

Code for vascular segmentation of large-scale cerebral two-photon microscopy angiograms.

**environment.yaml** lists dependencies used to run this code on a Nvidia Titan Xp GPU.


# Training new model
* The folder **Train_new_model/network** contains the code for training the network. 

* In order to train the model, the data can be downloaded from the following google drive link:
https://drive.google.com/open?id=1BIJFx8zs0IT1UX4AvgnHCj8k6dYh93o3

* Download the 'data' folder from the above link and copy it to the 'Train_new_model' folder, such that it's path is **.../Train_new_model/data**

* In the folder **Train_new_model/network**, execute the script main.py with default configurations as follows:

$ python main.py -d

# Using pretrained model for segmentation
* The folder **Test_trained_model** contains a pretrained model and code which can use that pretrained model to segment any preprocessed input angiogram from the user. 
 
* In order to perform segmentation on a sample 2PM angiogram (not used in the training process, and acquired on a different microscope than the data used for training the network), download the folder 'test_data' from the google drive link provided above, and copy it to the 'Test_trained_model' folder, such that it's path is **'../Test_trained_model/test_data'**. Not that the data in this folder has already been pre-processed using the method outlined in our paper [ref pending].

* In the folder **Test_trained_model**, execute the script main_test.py with default configurations as follows:

$ python main_test.py -d

* The model will segment all angiograms (in .mat format) in the 'test_data' folder and write the results to a new folder 'test_data_segmented'.




