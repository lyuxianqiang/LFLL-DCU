# LFLL-DCU
“Enhancing Low-light Light Field Images with A Deep Compensation Unfolding Network” 

# Dataset
You can download the dataset for synthetic LF data from 
https://drive.google.com/drive/folders/17_4TAdo3AN6qwZLosYCOFUd30evMDDwz?usp=sharing

The original link of L3F dataset seems invalid. Since I have graduated and left the university, I have not backed up the original PNG data. Therefore, I am providing the data after viewpoint extraction (central 7×7 viewpoints), which is stored in MAT format. You can get the data from
https://pan.baidu.com/s/1Pza7a2OrH2SwD9E9WizRGw
password: 44ch

# Requirements
- Python 3.8.8
- PyTorch 1.13.1

# Training
Set the training datapath, and learning rate according to data type. You can also change the batchsize accordingly. 

And run 'python train_synf.py'

When training on the L3F dataset, it is advisable to configure the learning rate to 5e-4 for L3F-20, 1e-4 for L3F-50, and 1e-4 for L3F-100, respectively. Or, you can download the modle from the link https://pan.baidu.com/s/13p3O7PXX0aK2Npg0U-SU2g 
with access code 'zwpy'

For other datasets, we suggest adapting the learning rate selection strategy based on the average brightness level of the dataset. Specifically, a lower learning rate is recommended for datasets with darker overall brightness.

# Testing
Set the testing datapath. 

And run 'python test_synf.py'
