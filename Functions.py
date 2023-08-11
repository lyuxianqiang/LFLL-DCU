from __future__ import print_function, division
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import warnings
from scipy import sparse
import random
import numpy as np

warnings.filterwarnings("ignore")
plt.ion()
def crop_boundary(I, crop_size):
    '''crop the boundary (the last 2 dimensions) of a tensor'''
    if crop_size == 0:
        return I

    if crop_size > 0:
        size = list(I.shape)
        I_crop = I.reshape(-1, size[-2], size[-1])
        I_crop = I_crop[:, crop_size:-crop_size, crop_size:-crop_size]
        size[-1] -= crop_size * 2
        size[-2] -= crop_size * 2
        I_crop = I_crop.reshape(size)
        return I_crop


#Initiate parameters in model 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0.0)

def SetupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def ExtractPatch(lf,lowlf, H, W, patchSize):
    indx=random.randrange(0,H-patchSize,8)
    indy=random.randrange(0,W-patchSize,8)

    lfPatch=lf[:,:,:,indx:indx+patchSize, indy:indy+patchSize]
    lowlfPatch=lowlf[:,:,:,indx:indx+patchSize, indy:indy+patchSize]
    return lfPatch,lowlfPatch #[u v c x y] 
def ExtractPatch_d(lf,lowlf, H, W, patchSize):
    indx=random.randrange(0,H-patchSize,8)
    indy=random.randrange(0,W-patchSize,8)

    lfPatch=lf[:,:,:,indx:indx+patchSize, indy:indy+patchSize]
    lowlfPatch=lowlf[:,:,:,indx:indx+patchSize, indy:indy+patchSize]
    return lfPatch,lowlfPatch #[u v c x y] 
    
def ResizeLF(lf,scale_factor):
    u,v,x,y,c=lf.shape
    resizedLF=np.zeros((u,v,int(scale_factor*x),int(scale_factor*y),c),dtype=np.int)
    for ind_u in range(u):
        for ind_v in range(v):
            view=lf[ind_u,ind_v,:,:,:]
            resizedView=cv2.resize(view, (int(scale_factor*x),int(scale_factor*y)), interpolation=cv2.INTER_CUBIC)
            resizedLF[ind_u,ind_v,:,:,:]=resizedView.reshape(int(scale_factor*x),int(scale_factor*y),-1)
    return resizedLF


def CropLF(lf,patchSize, overlap): #lf [b,u,v,c,x,y]
    b,u,v,c,x,y=lf.shape
    numX=0
    numY=0
    while (patchSize-overlap)*numX < x:
        numX = numX + 1 
    while (patchSize-overlap)*numY < y:
        numY = numY + 1 
    lfStack=torch.zeros(b,numX*numY,u,v,c,patchSize,patchSize)
    indCurrent=0
    for i in range(numX):
        for j in range(numY):
            if (i != numX-1)and(j != numY-1): 
                lfPatch=lf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
            elif (i != numX-1)and(j == numY-1): 
                lfPatch=lf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,-patchSize:]
            elif (i == numX-1)and(j != numY-1): 
                lfPatch=lf[:,:,:,:,-patchSize:,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
            else : 
                lfPatch=lf[:,:,:,:,-patchSize:,-patchSize:]
            # print(numX,numY,i,j,lfPatch.shape)
            lfStack[:,indCurrent,:,:,:,:,:]=lfPatch
            indCurrent=indCurrent+1
    return lfStack, [numX,numY] #lfStack [b,n,u,v,c,x,y] 

def Crop2LF(lowlf,normlf,lf,patchSize, overlap): #lf [b,u,v,c,x,y]
    b,u,v,c,x,y=lf.shape
    numX=0
    numY=0
    while (patchSize-overlap)*numX < x:
        numX = numX + 1 
    while (patchSize-overlap)*numY < y:
        numY = numY + 1 
    lfStack=torch.zeros(b,numX*numY,u,v,c,patchSize,patchSize)
    lowlfStack=torch.zeros(b,numX*numY,u,v,c,patchSize,patchSize)
    normStack=torch.zeros(b,numX*numY,u,v,c,patchSize,patchSize)
    indCurrent=0
    for i in range(numX):
        for j in range(numY):
            if (i != numX-1)and(j != numY-1): 
                lfPatch=lf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
                lowlfPatch=lowlf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
                normPatch=normlf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
            elif (i != numX-1)and(j == numY-1): 
                lfPatch=lf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,-patchSize:]
                lowlfPatch=lowlf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,-patchSize:]
                normPatch=normlf[:,:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,-patchSize:]
            elif (i == numX-1)and(j != numY-1): 
                lfPatch=lf[:,:,:,:,-patchSize:,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
                lowlfPatch=lowlf[:,:,:,:,-patchSize:,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
                normPatch=normlf[:,:,:,:,-patchSize:,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
            else : 
                lfPatch=lf[:,:,:,:,-patchSize:,-patchSize:]
                lowlfPatch=lowlf[:,:,:,:,-patchSize:,-patchSize:]
                normPatch=normlf[:,:,:,:,-patchSize:,-patchSize:]
            # print(numX,numY,i,j,lfPatch.shape)
            lfStack[:,indCurrent,:,:,:,:,:]=lfPatch
            lowlfStack[:,indCurrent,:,:,:,:,:]=lowlfPatch
            normStack[:,indCurrent,:,:,:,:,:]=normPatch
            indCurrent=indCurrent+1
    return lowlfStack, normStack, lfStack,[numX,numY] #lfStack [b,n,u,v,c,x,y] 

def Merge2LF(lfStack, coordinate, overlap, x, y):
    b,n,uv,c,patchSize,_=lfStack.shape
    lfMerged=torch.zeros(b,uv,c,x-overlap,y-overlap)
    for i in range(coordinate[0]):
        for j in range(coordinate[1]):
            if (i != coordinate[0]-1)and(j != coordinate[1]-1): 
                lfMerged[:,:,:,
                i*(patchSize-overlap):(i+1)*(patchSize-overlap),
                j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,
                                                                            overlap//2:-overlap//2,
                                                                            overlap//2:-overlap//2] 
            elif (i == coordinate[0]-1)and(j != coordinate[1]-1): 
                lfMerged[:,:,:,i*(patchSize-overlap):,
                                 j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,
                                                                                            -((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
                                                                                            overlap//2:-overlap//2]            
            elif (i != coordinate[0]-1)and(j == coordinate[1]-1): 
                lfMerged[:,:,:,i*(patchSize-overlap):(i+1)*(patchSize-overlap),
                                 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,
                                                                    overlap//2:-overlap//2,
                                                                    -((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]
            else: 
                lfMerged[:,:,:,i*(patchSize-overlap):,
                                 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,
                                                                    -((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
                                                                    -((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]    
    return lfMerged # [b,u,v,c,x,y]

def MergeLF(lfStack, coordinate, overlap, x, y):
    b,n,u,v,c,patchSize,_=lfStack.shape
    lfMerged=torch.zeros(b,u,v,c,x-overlap,y-overlap)
    for i in range(coordinate[0]):
        for j in range(coordinate[1]):
            if (i != coordinate[0]-1)and(j != coordinate[1]-1): 
                lfMerged[:,:,:,:,
                i*(patchSize-overlap):(i+1)*(patchSize-overlap),
                j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,:,
                                                                            overlap//2:-overlap//2,
                                                                            overlap//2:-overlap//2] 
            elif (i == coordinate[0]-1)and(j != coordinate[1]-1): 
                lfMerged[:,:,:,:,i*(patchSize-overlap):,
                                 j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,:,
                                                                                            -((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
                                                                                            overlap//2:-overlap//2]            
            elif (i != coordinate[0]-1)and(j == coordinate[1]-1): 
                lfMerged[:,:,:,:,i*(patchSize-overlap):(i+1)*(patchSize-overlap),
                                 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,:,
                                                                    overlap//2:-overlap//2,
                                                                    -((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]
            else: 
                lfMerged[:,:,:,:,i*(patchSize-overlap):,
                                 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,:,
                                                                    -((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
                                                                    -((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]    
    return lfMerged # [b,u,v,c,x,y]

def ComptPSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)
