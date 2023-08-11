import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import math
import numpy as np

class Conv_spa(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_spa, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):     # [N,32,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.reshape(N*u*v,c,h,w)  # [N*uv,32,h,w]
        out = self.op(x)
        #print(out.shape)
        out = out.reshape(N,u,v,32,h,w)
        return out

class Conv_ang(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, angular, bias):
        super(Conv_ang, self).__init__()
        self.angular = angular
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):    # [N,32,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.permute(0,4,5,3,1,2).reshape(N*h*w,c,self.angular,self.angular)   #[N*h*w,32,7,7]
        out = self.op(x)
        out = out.reshape(N,h,w,32,u,v).permute(0,4,5,3,1,2)
        return out

    
    
class Conv_epi_h(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_h, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):    # [N,64,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.permute(0,1,4,3,2,5).reshape(N*u*h,c,v,w)
        out = self.op(x)
        out = out.reshape(N,u,h,32,v,w).permute(0,1,4,3,2,5)
        return out

class Conv_epi_v(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_v, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        N,u,v,c,h,w = x.shape
        x = x.permute(0,2,5,3,1,4).reshape(N*v*w,c,u,h)
        out = self.op(x)
        out = out.reshape(N,v,w,32,u,h).permute(0,4,1,3,5,2)
        return out

class Autocovnlayer(nn.Module):
    def __init__(self,dence_num,component_num,angular,fn,bs):
        super(Autocovnlayer, self).__init__()
        self.dence_num = dence_num
        self.component_num = component_num
        self.angular = angular
        self.kernel_size = 3

        self.naslayers = nn.ModuleList([
           Conv_spa(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_ang(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
           Conv_epi_h(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_epi_v(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
        ])
        ###################
        self.epi_boost = nn.Conv2d(in_channels = fn, out_channels=fn, kernel_size=3, stride=1, padding=1, bias = bs)
        self.Conv_all = nn.Conv2d(in_channels = fn+4, out_channels=fn, kernel_size=3, stride=1, padding=1, bias = bs)
        self.Conv_mixray = nn.Conv2d(in_channels = angular*angular, out_channels=angular*angular, kernel_size=3, stride=1, padding=1, bias = True)
        self.Conv_down = nn.Conv2d(in_channels = 32, out_channels=4, kernel_size=1, stride=1, padding=0, bias = False)
        self.Conv_mixdence = nn.Conv2d(in_channels = fn*self.dence_num, out_channels=fn, kernel_size=1, stride=1, padding=0, bias = False)
        self.Conv_mixnas = nn.Conv2d(in_channels = 32*5, out_channels=fn, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        x = torch.stack(x,dim = 0) # [dence_num N  C  uv H W]  [dence_num,N,64,uv,h,w]
        [fn, N, u,v, C, h, w] = x.shape 
        x = x.permute([1,2,3,0,4,5,6]).reshape([N*u*v,fn*C,h,w])    # [N*uv, fn*c, h,w] 
        x = self.relu(self.Conv_mixdence(x))                               # ==> [N*uv, c', h, w] 
        x_mix = x.reshape([N,u,v,C,h,w])   # [N,64,uv,h,w] !!!
        nas = []
        for layer in self.naslayers:
            nas_ = layer(x_mix)
            nas.append(nas_)
#         print(nas[-1].shape)
        x_epi = nas[-1] + nas[-2]    # (N,uv,32,h,w)
        nas_ = self.relu(self.epi_boost(x_epi.reshape(N*u*v,C,h,w)))
        nas.append(nas_.reshape(N,u,v,C,h,w))
        nas_1 = self.relu(self.Conv_mixray(x_mix.permute([0,3,1,2,4,5]).reshape(N*32,u*v,h,w)))
        nas_1 = nas_1.reshape(N,32,u,v,h,w).permute([0,2,3,1,4,5])
        nas_1 = self.relu(self.Conv_down(nas_1.reshape(N*u*v,32,h,w)))
        
        nas = torch.stack(nas,dim = 0)   
        nas = nas.permute([1,2,3,0,4,5,6]).reshape([N*u*v,5*32,h,w])   ##[N*uv, fn*c, h,w] 
        nas = self.relu(self.Conv_mixnas(nas))           

        nas_2 = self.Conv_all(torch.cat([nas,nas_1],dim=1))
        nas_2 = nas_2.reshape(N,u,v,C,h,w)
        out = self.relu(x_mix + nas_2)
        return out

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()


class Illumination(nn.Module):
    def __init__(self,alpha,opt):
        super(Illumination, self).__init__()
        self.alpha = alpha
        self.maxpoll = nn.MaxPool1d(kernel_size=3,stride=1,padding=0)
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,bias = True)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1,bias = True)
        self.dense4d = make_autolayers(opt.sasLayerNum,32,opt)
        self.gamma = nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lowlf):
        N,u,v,c,h,w = x.shape
        I_init = x - self.gamma * (x - lowlf)
        I_init = self.maxpoll(I_init.reshape(N*u*v,c,h*w).permute([0,2,1]))   # out [Nuv,hw,1]
        I_init = I_init.permute([0,2,1]).reshape(N*u*v,1,h,w)
        I_feat = self.relu(self.conv0(I_init))
        I_feat = I_feat.reshape(N,u,v,32,h,w)
        feat = [I_feat]
        for index,layer in enumerate(self.dense4d):
            feat_ = layer(feat)
            feat.append(feat_)
        out = I_init.expand(-1,c,-1,-1) - self.conv1(feat[-1].reshape(N*u*v,32,h,w))
        return out.reshape(N,u,v,c,h,w)


class StageBlock(nn.Module):
    def __init__(self, opt):
        super(StageBlock,self).__init__()
        self.illum = Illumination(1,opt)
        self.gredient_conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,bias = False)
        self.gredient_conv1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1,bias = False)
        self.gredient_dense = make_autolayers(opt.sasLayerNum,32,opt)

        self.denoise_conv0 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1,bias = True)
        self.denoise_conv1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1,bias = True)
        self.denosie_dense = make_autolayers(opt.sasLayerNum,32,opt)

        self.relu = nn.ReLU(inplace=True)
        self.delta=nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda()) 
        self.eta=nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        
    def forward(self,out_lastStage,imap_lastStage,lowlf,idx):
        N,u,v,c,h,w= out_lastStage.shape
        if idx == 0:
            n_in = torch.cat([out_lastStage,lowlf],dim=3)
            n_feat = self.relu(self.denoise_conv0(n_in.reshape(N*u*v,2*c,h,w)))
        else:
            n_in = lowlf - out_lastStage * torch.mean(lowlf,[1,2,3,4,5],keepdim=True) / torch.mean(out_lastStage,[1,2,3,4,5],keepdim=True)
            n_in = torch.cat([n_in,lowlf],dim=3)
            n_feat = self.relu(self.denoise_conv0(n_in.reshape(N*u*v,2*c,h,w)))  
         
        n_feat = n_feat.reshape(N,u,v,32,h,w)
        feat2 = [n_feat]
        for idx, layer2 in enumerate(self.denosie_dense):
            feat_ = layer2(feat2)
            feat2.append(feat_)
        out_denoise = lowlf - self.denoise_conv1(feat2[-1].reshape(N*u*v,32,h,w)).reshape(N,u,v,c,h,w)

        imap = self.illum(out_lastStage,lowlf)
        err1 = imap * (imap * out_lastStage - out_denoise)
        err2_feat = self.relu(self.gredient_conv0(out_lastStage.reshape(N*u*v,c,h,w)))
        err2_feat = err2_feat.reshape(N,u,v,32,h,w)
        feat = [err2_feat]
        for index,layer in enumerate(self.gredient_dense):
            feat_ = layer(feat)
            feat.append(feat_)
        err2 = self.gredient_conv1(feat[-1].reshape(N*u*v,32,h,w))
        out_currentStage = out_lastStage - self.delta * (err1 + self.eta * err2.reshape(N,u,v,c,h,w))
        return out_currentStage,imap
        
        
def CascadeStages(block, opt):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(opt))
    return blocks

def make_autolayers(LayerNum,fn,opt):
    layers = []
    for i in range(LayerNum):
        layers.append(Autocovnlayer(i+1, 4, opt.angResolution, fn, True))
    return nn.Sequential(*layers)

class Main_unrolling(nn.Module):
    def __init__(self,opt):
        super(Main_unrolling, self).__init__()
        # Iterative stages
        self.iterativeRecon = CascadeStages(StageBlock, opt)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 

    def forward(self,lowlf):
        out = lowlf
        imap = torch.ones(lowlf.shape).cuda()
        for idx, stage in enumerate(self.iterativeRecon):
            out,imap = stage(out,imap,lowlf,idx)
        return out


