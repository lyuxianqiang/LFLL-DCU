import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""

    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = scio.loadmat(opt.dataPath)    
        self.lfSet = dataSet['lf'].transpose(0,1,4,2,3) 
        self.lowlfSet = dataSet['lowlf_noise'].transpose(0,1,4,2,3) 
        self.patchSize=opt.patchSize

    def __getitem__(self, idx):

        lf=self.lfSet 
        lowlf = self.lowlfSet 
#         lfPatch, lowlfPatch=ExtractPatch(lf, lowlf, H, W, self.patchSize) #[u v c x y]
        lfPatch= torch.from_numpy(lf.astype(np.float32)/255)
        lowlfPatch= torch.from_numpy(lowlf.astype(np.float32)/255)
        
        sample = {'lf':lfPatch,'lowlf':lowlfPatch}
        return sample
        
    def __len__(self):
        return 1



