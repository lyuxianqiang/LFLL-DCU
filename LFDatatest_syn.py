import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""

    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = scio.loadmat(opt.dataPath)    
        self.lfSet = dataSet['lf'].transpose(5,0,1,4,2,3) #[ind, u, v, c, x, y]
        self.lowlfSet = dataSet['lowlf_noise'].transpose(5,0,1,4,2,3) #[ind, u, v, c, x, y]
        self.lfNameSet = dataSet['LF_name']
        self.patchSize=opt.patchSize

    def __getitem__(self, idx):
        lf=self.lfSet[idx] 
        lowlf = self.lowlfSet[idx] 
        lfPatch= torch.from_numpy(lf.astype(np.float32)/255)
        lowlfPatch= torch.from_numpy(lowlf.astype(np.float32)/255)
        LF_name = ''.join([chr(self.lfNameSet[idx][0][0][i]) for i in range(self.lfNameSet[idx][0][0].shape[0])]) 
        sample = {'lf':lfPatch,'lowlf':lowlfPatch,'lfname':LF_name}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]



