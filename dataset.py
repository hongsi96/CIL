import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import pdb
import numpy as np
import random


class custom_CIFAR(torchvision.datasets.CIFAR100):
    def seperate(self, class_num):
        self.c_data=[]
        self.c_targets=[]
        for i in range(class_num):
            idxes=np.where(np.array(self.targets)==i)[0]
            self.c_data.append(self.data[idxes])
            if self.train:
                self.c_targets.append([i]*500)
            else:
                self.c_targets.append([i]*100)
        self.data=[]
        self.targets=[]
        self.memory={}
        self.tasks=10
    def select(self,cls):
        self.data=[]
        self.targets=[]
        for c in cls:
            self.data.extend(self.c_data[c])
            self.targets.extend(self.c_targets[c])

        if self.train and len(self.memory) >0:
            count=2000//len(self.memory)
            #for mem in self.memory:
            for i, key in enumerate(self.memory.keys()):
                #key : class label
                self.data.extend(self.c_data[key][self.memory[key][:count]])
                self.targets.extend(self.c_targets[key][:count])

        self.data=np.array(self.data)
        self.targets=np.array(self.targets)

    def sampling(self,cls, mode='random',model=None):
        if mode=='random':
            for c in cls: 
                idx=random.sample(range(500), 500)
                self.memory[c]=idx
        elif mode=='herding' and model is not None:
            for c in cls:
                if isinstance(model, nn.DataParallel):
                    features=model.module.extract(torch.tensor(self.c_data[c].transpose(0,3,1,2)).float().cuda().detach())
                else:
                    features=model.extract(torch.tensor(self.c_data[c].transpose(0,3,1,2)).float().cuda().detach())
                mean=features.mean(dim=0)
                mean=F.normalize(mean,dim=0)
                features=F.normalize(features,dim=1)
                distances=torch.pow(mean-features,2).sum(-1)
                idx=distances.argsort().cpu().numpy()

                self.memory[c]=idx.tolist()
        else:
            pdb.set_trace()
    







