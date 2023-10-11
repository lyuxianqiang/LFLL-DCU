import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from LFDataset_syn import LFDataset
from Functions import weights_init,SetupSeed,CropLF, MergeLF,ComptPSNR,rgb2ycbcr
from DeviceParameters import to_device
import itertools,argparse
import numpy as np
import scipy.io as scio 
import scipy.misc as scim

from torchvision.models.vgg import vgg16
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
from collections import defaultdict
from os.path import join
from MainNet_unrolling_all import Main_unrolling

# Training settings
parser = argparse.ArgumentParser(description="Light Field Compressed Sensing")
parser.add_argument("--learningRate", type=float, default=1e-3, help="Learning rate, recommend for syn lr=1e-3 and for l3fdata lr=1e-4.")
parser.add_argument("--step", type=int, default=1000, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--sasLayerNum", type=int, default=6, help="The number of stages")
parser.add_argument("--batchSize", type=int, default=2, help="Batch size, changing according the GPU's memory")
parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--num_cp", type=int, default=1000, help="Number of epoches for saving checkpoint")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")
parser.add_argument("--epochNum", type=int, default=10010, help="The number of epoches")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")
parser.add_argument("--summaryPath", type=str, default='./', help="Path for saving training log ")
parser.add_argument("--dataName", type=str, default='Synthetic', help="The name of dataset ")
parser.add_argument("--dataPath", type=str, default='***training_data.mat***', help="Path for loading training data ")
parser.add_argument("--augment", type=bool, default=False, help="Whether to perform data augmentation")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
opt = parser.parse_known_args()[0]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Training_lfll_{}_{}_{}_{}_{}.log'.format(opt.dataName, opt.stageNum, opt.sasLayerNum, opt.epochNum, opt.learningRate))
log.addHandler(fh)
logging.info(opt)


class perception_loss(torch.nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = torch.nn.Sequential() 
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        h1 = self.to_relu_1_2(x1)
        h2 = self.to_relu_1_2(x2)
        p_loss = torch.nn.functional.mse_loss(h1,h2)
        return p_loss



if __name__ == '__main__':
    SetupSeed(50)
    savePath = './model/lfll_unrolling_{}_{}_{}_{}_{}'.format(opt.dataName, opt.stageNum, opt.sasLayerNum, opt.epochNum, opt.learningRate)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    lfDataset = LFDataset(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batchSize,shuffle=True,num_workers=8)
    print('loaded {} LFIs from {}'.format(len(dataloader), opt.dataPath))

    model=Main_unrolling(opt)
    model = model.cuda()
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training parameters: %d" %total_trainable_params)

    criterion = torch.nn.L1Loss()
    ploss = perception_loss().cuda()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate) #optimizer
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = opt.learningRate,steps_per_epoch=len(dataloader),epochs=opt.epochNum,pct_start = 0.2, div_factor = 10, final_div_factor = 10)

    if opt.resume_epoch:
        resume_path = join(savePath,'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))   


    lossLogger = defaultdict(list)
    for epoch in range(opt.resume_epoch+1,opt.epochNum):
        batch = 0
        lossSum = 0
        for _,sample in enumerate(dataloader):
            batch = batch +1
            lf=sample['lf']    #[b,u v c x y] 
            lf = lf.cuda()
            lowlf=sample['lowlf'].cuda()
            estimatedLF=model(lowlf)
            
            loss1 = criterion(estimatedLF,lf)
            loss2 = ploss(estimatedLF.reshape(-1,3,32,32),lf.reshape(-1,3,32,32))
            loss3 = 1 - ssim(torch.clamp(estimatedLF.reshape(-1,3,32,32),min=0.0, max=1.0),lf.reshape(-1,3,32,32),data_range=1.0,size_average=True)
            loss = loss1 + loss2 + 0.1 * loss3
            lossSum += loss.item()
            print("Epoch: %d Batch: %d Loss: %.6f" %(epoch,batch,loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()     #ONE

        if epoch % opt.num_cp == 0:
            model_save_path = join(savePath,"model_epoch_{}.pth".format(epoch))
            state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}
            torch.save(state,model_save_path)
            print("checkpoint saved to {}".format(model_save_path))
        log.info("Epoch: %d Loss: %.6f" %(epoch,lossSum/len(dataloader)))

        #Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(lossSum/len(dataloader))
        lossLogger['Lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])

    # plt.figure()
    # plt.title('Loss')
    # plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
    # plt.savefig('Training_lfll_{}_{}_{}_{}_{}_ver.jpg'.format(opt.dataName,opt.stageNum,opt.sasLayerNum, opt.epochNum, opt.learningRate))
    # plt.close()











