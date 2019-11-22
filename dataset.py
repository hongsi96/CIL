import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import pdb
import numpy as np
import random
from PIL import Image


class custom_CIFAR(torchvision.datasets.CIFAR100):
    def seperate(self, class_num, memory=2000):
        self.num_mem=memory
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
        self.grad_set=np.zeros((100, 500))
    def select(self,cls):
        self.current_class=len(cls)
        self.data=[]
        self.targets=[]
        for c in cls:
            self.data.extend(self.c_data[c])
            self.targets.extend(self.c_targets[c])

        if self.train and len(self.memory)>0 and self.num_mem>0:
            count=self.num_mem//len(self.memory)
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
        elif mode=='attention':
            for c in cls:
                data=torch.tensor(self.c_data[c].transpose(0,3,1,2)).float().cuda().detach()
                order=model.module.getout(data,self.current_class//10-1,sampling=True)
                order=order.view(-1).cpu().detach().numpy()
                order=order.argsort()[::-1]
                self.memory[c]=order.tolist()
        elif mode=='herding' and model is not None:
            for c in cls:
                if isinstance(model, nn.DataParallel):
                    #pdb.set_trace()
                    features=model.module.extract(torch.tensor(self.c_data[c].transpose(0,3,1,2)).float().cuda().detach())
                else:
                    features=model.extract(torch.tensor(self.c_data[c].transpose(0,3,1,2)).float().cuda().detach())
                mean=features.mean(dim=0)
                #mean=F.normalize(mean,dim=0)
                #features=F.normalize(features,dim=1)
                distances=torch.pow(mean-features,2).sum(-1)
                idx=distances.argsort().cpu().numpy()

                self.memory[c]=idx.tolist()#[::-1]
        else:
            pdb.set_trace()
    def grad_storage(self, indexes, grades, labeles):
        #grad_set
        for index, grad, label in zip(indexes, grades, labeles):
            if index >-1:
                self.grad_set[label][index]+=grad.sum()

    def __getitem__(self, index):
        img, target =self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            if index>=500*self.current_class:
                index=-1
            else:
                index=index%500
        else:
            if index>=100*self.current_class:
                index=-1
            else:
                index=index%100
        return index, img, target
    







