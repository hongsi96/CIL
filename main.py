from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pdb
import torchvision
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import models
import dataset
import copy
import logging
import os
from tensorboardX import SummaryWriter
import utils
from tqdm import tqdm

filename='test_distil_herding'
init_lr=0.1
momentum=0.9
wd=5e-4
if not os.path.exists('experiments'):
    os.makedirs('experiments')

logging.basicConfig(filename='./experiments/{}.txt'.format(filename), level=logging.DEBUG)

def adjust_learning_rate(optimizer,epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            #print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
            param_group['lr'] *= 0.1
def get_performance(output, target):
    acc = (output.max(1)[1] == target).to(torch.float).mean()
    return acc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=63/255, contrast=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323,0.48654887331495095,0.4409178433670343), (0.2673342858792409, 0.25643846291708816, 0.2761504713256834)),])

transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323,0.48654887331495095,0.4409178433670343), (0.2673342858792409, 0.25643846291708816, 0.2761504713256834)),
            ])

#tasks_c=random.sample(range(100),100)
tasks_c=[i for i in range(100)]
tasks=[]
for i in range(10):
    tasks.append(tasks_c[i*10:(i+1)*10])

writer = SummaryWriter(log_dir='log')
logging.info(tasks)
model = models.resnet()
model = torch.nn.DataParallel(model).cuda() 

cifar_train=dataset.custom_CIFAR('data', train=True, transform=transform_train, target_transform=None, download=True)
cifar_test=dataset.custom_CIFAR('data', train=False, transform=transform_test, target_transform=None, download=True)
cifar_train.seperate(class_num=100); cifar_test.seperate(class_num=100)

model_classifier=model.module.classifier if isinstance(model, nn.DataParallel) else model.classifier
seen=[]
for n_task, task in enumerate(tasks):
    seen.extend(task)
    model_classifier.set_trainable(seen)
    if n_task==0:
        cifar_train.select(cls=task); cifar_test.select(cls=seen)
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1).cuda()
        train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128,shuffle=True, num_workers=8)
        test_loader=torch.utils.data.DataLoader(cifar_test, batch_size=100,shuffle=True, num_workers=8)
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)

        for epoch in range(200):
            adjust_learning_rate(optimizer,epoch, schedule=[120, 160, 180])
            model.train()
            loss_train=0;loss_test=0
            loss_ = utils.AverageMeter();acc_  = utils.AverageMeter()
            pbar  = tqdm(train_loader, desc='T:{}, train {:d}'.format(n_task,epoch), ascii=True, ncols=80)
            for data, label in pbar:#for ii, (data, label) in enumerate(train_loader):
                data=data.cuda(); label=label.cuda()
                optimizer.zero_grad()
                output=model(data)
                 
                loss=criterion(output[:,task], label).mean()
                loss.backward(); optimizer.step()

                acc = get_performance(output[:,task], label)
                loss_.update((loss.mean()).item(), data.size(0))
                acc_.update(acc.item(), data.size(0))
                pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))
            
            #test
            if epoch%10==0:
                model.eval()
                loss_ = utils.AverageMeter();acc_  = utils.AverageMeter()
                pbar  = tqdm(test_loader, desc='T:{}, test {:d}'.format(n_task,epoch), ascii=True, ncols=80)
                with torch.no_grad():
                    for data, label in pbar:
                        data=data.cuda(); label=label.cuda() 
                        output=model(data) 
                        loss=criterion(output[:,task], label).mean()
                        #loss_test+=loss.sum()
                        acc = get_performance(output[:,task], label)
                        loss_.update(loss.item(), data.size(0))
                        acc_.update(acc.item(), data.size(0))
                        pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))

            #writer.add_scalar('Train/task_{}/loss'.format(n_task), loss_train.item()/len(cifar_train), epoch)
            
        torch.save(model.state_dict(), 'save/model_{}.t7'.format(n_task))
        cifar_train.sampling(cls=task, mode='herding', model=model)
        prev_model=copy.deepcopy(model)
    else:
        #training
        cifar_train.select(cls=task);cifar_test.select(cls=seen)
        criterion={}
        criterion.update({'cls': nn.CrossEntropyLoss(reduction='none', ignore_index=-1).to(device)})
        criterion.update({'dst': nn.KLDivLoss(reduction='none').to(device)})

        train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128,shuffle=True, num_workers=8)
        test_loader=torch.utils.data.DataLoader(cifar_test, batch_size=100,shuffle=True, num_workers=8)
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd)
        for epoch in range(180):
            adjust_learning_rate(optimizer,epoch, schedule=[120, 160, 170])
            model.train()
            cls_loss_train=0;dst_loss_train=0
            loss_ = utils.AverageMeter();acc_  = utils.AverageMeter()
            pbar  = tqdm(train_loader, desc='T:{}, train {:d}'.format(n_task,epoch), ascii=True, ncols=80)
            for data, label in pbar:
                data=data.cuda();label=label.cuda() 
                
                optimizer.zero_grad() 
                output=model(data);prev_output=prev_model(data)
                #cls
                loss_c=criterion['cls'](output[:,seen], label).mean()
                #dist 
                output_pld=(output[:,seen[:-10]] / 2).log_softmax(dim=1)
                starget_pld = (prev_output[:,seen[:-10]] / 2).softmax(dim=1)
                #output_pld=(output[:,seen] / 2).log_softmax(dim=1)
                #starget_pld = (prev_output[:,seen] / 2).softmax(dim=1)
                loss_d=(criterion['dst'](output_pld, starget_pld)).sum(dim=1).mean()* (2**2)

                loss=loss_c+loss_d
                loss.backward();optimizer.step() 
                acc = get_performance(output[:,seen], label) 
                loss_.update(loss.item(), data.size(0))
                acc_.update(acc.item(), data.size(0))
                pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))
            #writer.add_scalar('Train/task_{}/loss'.format(n_task), loss.item(), epoch)
        cifar_train.sampling(cls=task, mode='herding', model=model)#cifar_train.sampling(cls=task)
        #finetuning
        cifar_train.select(cls=[])
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1).cuda()
        print('Finetuning, memory:{}'.format(len(cifar_train)))
        train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128,shuffle=True, num_workers=8)
        test_loader=torch.utils.data.DataLoader(cifar_test, batch_size=100,shuffle=True, num_workers=8) 
        optimizer = optim.SGD(model_classifier.parameters(), lr=init_lr*0.1, momentum=momentum, weight_decay=wd)
        for epoch in range(20):
            adjust_learning_rate(optimizer,epoch, schedule=[10,15])
            loss_train=0;loss_test=0
            model.train()
            loss_ = utils.AverageMeter();acc_  = utils.AverageMeter()
            pbar  = tqdm(train_loader, desc='F:{}, train {:d}'.format(n_task,epoch), ascii=True, ncols=80)
            for data, label in pbar: 
                data=data.cuda();label=label.cuda()
                optimizer.zero_grad() 
                output=model(data)
                loss=criterion(output[:,seen], label).mean()
                loss.backward();optimizer.step()  

                acc = get_performance(output[:,seen], label)
                loss_.update(loss.item(), data.size(0))
                acc_.update(acc.item(), data.size(0))
                pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg)) 

            model.eval()
            loss_ = utils.AverageMeter();acc_  = utils.AverageMeter()
            pbar  = tqdm(test_loader, desc='F:{}, test {:d}'.format(n_task,epoch), ascii=True, ncols=80)
            with torch.no_grad():
                for data, label in pbar:
                    data=data.cuda();label=label.cuda()
                    output=model(data)
                    loss=criterion(output[:,seen], label).mean()

                    acc = get_performance(output[:,seen], label) 
                    loss_.update(loss.item(), data.size(0))
                    acc_.update(acc.item(), data.size(0))
                    pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg)) 
            
            #writer.add_scalar('Train/task_{}/loss'.format(n_task), loss.item(), epoch)
        torch.save(model.state_dict(), 'save/model_{}.t7'.format(n_task))
        prev_model=copy.deepcopy(model)
    #final performance   
    perform=[]
    for tt in range(n_task+1):
        cifar_test.select(seen[tt*10:tt*10+10])    
        test_loader=torch.utils.data.DataLoader(cifar_test, batch_size=100,shuffle=True, num_workers=8)
        for ii, (data, label) in enumerate(test_loader):
            data=data.cuda();label=label.cuda()
            output=model(data)
            acc = get_performance(output[:,seen], label)
            acc_.update(acc.item(), data.size(0))
        perform.append(acc_.avg)
    logging.info(perform)  

        


