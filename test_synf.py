import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging,argparse
import warnings
from LFDatatest_syn import LFDataset
from Functions import weights_init,SetupSeed,CropLF, MergeLF,ComptPSNR,rgb2ycbcr
import itertools,argparse
from skimage.metrics import structural_similarity
import lpips

import numpy as np
import scipy.io as scio 
import matplotlib.pyplot as plt
from collections import defaultdict
from os.path import join
from MainNet_unrolling_all import Main_unrolling


# test settings
parser = argparse.ArgumentParser(description="AUNS-LL")
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--sasLayerNum", type=int, default=6, help="The number of layers")
parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--patchSize", type=int, default=256, help="The size of croped LF patch")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")
parser.add_argument("--summaryPath", type=str, default='./', help="Path for saving training log ")
parser.add_argument("--dataName", type=str, default='Synthetic', help="The name of dataset ")
parser.add_argument("--modelPath", type=str, default='./model/model_synf_simple.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='./input/****.mat', help="Path for test data ")
parser.add_argument("--savePath", type=str, default='./results/', help="Path for saving results ")
opt = parser.parse_known_args()[0]

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Testing_original.log')
log.addHandler(fh)

lf_dataset = LFDataset(opt)
dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize,shuffle=False)
print('loaded {} LFIs from {}'.format(len(dataloader), opt.dataPath))

loss_lpips_alex = lpips.LPIPS(net='alex')
model=Main_unrolling(opt)
model.load_state_dict(torch.load(opt.modelPath)['model'])
model.eval()
model.cuda()

with torch.no_grad():
#     SetupSeed(50)
    num = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_time = 0
    for _,sample in enumerate(dataloader):
        num=num+1
        LF=sample['lf'] #test lf [b,u,v,x,y,c]
        lowlf=sample['lowlf']
        lowlf = lowlf.cuda()
        lfName=sample['lfname']
        b,u,v,c,x,y = LF.shape   
        print(LF.shape)
        ## Crop the input LF into patches 
#         LFStack,coordinate=CropLF(lowlf,opt.patchSize, opt.overlap) #[b,n,u,v,c,x,y]
#         n=LFStack.shape[1]       
#         print(LFStack.shape)
#         estiLFStack=torch.zeros(b,n,u,v,c,opt.patchSize,opt.patchSize)#[b,n,u,v,c,x,y]
#         # reconstruction
#         for i in range(LFStack.shape[1]):
#             estiLFStack[:,i,:,:,:,:,:] = model(LFStack[:,i,:,:,:,:,:].cuda())
#         estiLF=MergeLF(estiLFStack,coordinate,opt.overlap,x,y) #[b,u,v,c,x,y]
#         b,u,v,c,xCrop,yCrop=estiLF.shape
#         estiLF = torch.clamp(estiLF,min=0.0, max=1.0)
#         LF=LF[:,:,:,:, opt.overlap//2:opt.overlap//2+xCrop,opt.overlap//2:opt.overlap//2+yCrop]
        estiLF = model(lowlf)
        estiLF = torch.clamp(estiLF,min=0.0, max=1.0)
        b,u,v,c,xCrop,yCrop=estiLF.shape
        
        lf_psnr = 0
        lf_ssim = 0
        lf_lpips = 0
        #evaluation
        for ind_uv in range(u*v):
                lf_psnr += ComptPSNR(estiLF.reshape(b,u*v,c,xCrop,yCrop)[0,ind_uv].cpu().numpy(),LF.reshape(b,u*v,c,xCrop,yCrop)[0,ind_uv].cpu().numpy())  / (u*v)
                lf_ssim += structural_similarity((estiLF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8), (LF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8),gaussian_weights=True,sigma=1.5,use_sample_covariance=False,multichannel=True) / (u*v)
                lf_lpips += loss_lpips_alex(estiLF.reshape(b,u*v,c,xCrop,yCrop)[0,ind_uv].cpu(),LF.reshape(b,u*v,c,xCrop,yCrop)[0,ind_uv]) / (u*v)

        avg_psnr += lf_psnr / len(dataloader)             
        avg_ssim += lf_ssim / len(dataloader)
        avg_lpips += lf_lpips / len(dataloader)
        log.info('Index: %d  Scene: %s  PSNR: %.2f  SSIM: %.3f   LPIPS:  %.3F'%(num,lfName[0],lf_psnr,lf_ssim,lf_lpips))
        #save reconstructed LF
        scio.savemat(os.path.join(opt.savePath,lfName[0]+'.mat'),{'lf_recons':torch.squeeze(estiLF).cpu().numpy()}) #[u,v,x,y,c]

    [log.info('Average PSNR: %.2f  SSIM: %.3f  LPIPS: %.3f '%(avg_psnr,avg_ssim,avg_lpips))  ]










