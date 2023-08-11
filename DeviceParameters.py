from __future__ import print_function, division
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.ion()

#Wrap a dataloader to move data to a device
class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)


#Move tensor(s) to chosen device
def to_device(data, device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)