import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch_d

class LFDataset(Dataset):
    """Light Field dataset."""
    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = scio.loadmat(opt.dataPath)   
        self.lfSet = dataSet['lf'].transpose(5,0,1,4,2,3) 
        self.lowlfSet = dataSet['lowlf_noise'].transpose(5,0,1,4,2,3) #[ind, u, v, c, x, y]
        self.patchSize=opt.patchSize


    def __getitem__(self, idx):
        
        lf=self.lfSet[idx]
        lf = np.array(lf, dtype="float32") / 255.0
        lowlf = self.lowlfSet[idx]
        lowlf = np.array(lowlf, dtype="float32") / 255.0
        H = self.lfSet.shape[4]
        W = self.lfSet.shape[5]

        lfPatch, lowlfPatch =ExtractPatch_d(lf, lowlf, H, W, self.patchSize) #[u v c x y]
        lfPatch= torch.from_numpy(lfPatch)
        lowlfPatch= torch.from_numpy(lowlfPatch)
        
        sample = {'lf':lfPatch,'lowlf':lowlfPatch}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]

