from __future__ import print_function
from site import addusersitepackages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from util import *

import os
import sys
import time
import argparse
import datetime
import pandas as pd

from networks.wide_resnet import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')

parser.add_argument('--method', default="baseline", type=str, help='cutout/baseline/mixup/cutmix')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--batch_size', default=128, type=int, help='num of batch_size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--num_epochs', default=200, type=int, help='num of epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--seed', default=980038, type=int, help='dropout_rate')

args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
batch_size = args.batch_size
num_epochs = args.num_epochs

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

####设置随机种子
setup_seed(args.seed)
print("set seed seuccess!")

if args.method=="baseline":
    writer = SummaryWriter("runs"+os.sep+"baseline")
elif args.method=="cutmix":
    writer = SummaryWriter("runs"+os.sep+"cutmix")
elif args.method=="mixup":
    writer = SummaryWriter("runs"+os.sep+"mixup")
elif args.method=="cutout":
    transform_train.transforms.append(Cutout(n_holes=1, length=8))
    writer = SummaryWriter("runs"+os.sep+"cutout")

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

net = Wide_ResNet(args.depth,num_classes,args.widen_factor, args.dropout)

# Test only option
if (args.testOnly):
    checkpoint = torch.load('./model_data/'+args.dataset + os.sep + args.method+'.t7')
    net = checkpoint['net']

    if use_cuda: 
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    net.training = True
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=args.momentum, weight_decay=args.weight_decay)
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    if args.method=="baseline":
        return baseline(trainloader,net,criterion,len(trainset),batch_size,num_epochs,epoch,optimizer)
    elif args.method=="cutmix":
        return cutmix(trainloader,net,criterion,len(trainset),batch_size,num_epochs,epoch,optimizer)
    elif args.method=="mixup":
        return mixup(trainloader,net,criterion,len(trainset),batch_size,num_epochs,epoch,optimizer)
    elif args.method=="cutout":
        return cutout(trainloader,net,criterion,len(trainset),batch_size,num_epochs,epoch,optimizer)

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+args.method+'.t7')
            best_acc = acc
        return (float(test_loss/(1+batch_idx)), float(correct/total))

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0

now_time = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
log = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc','test_loss','test_acc'])
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train_loss,train_acc= train(epoch)
    test_loss,test_acc = test(epoch)
    writer.add_scalar('Train/loss', train_loss, epoch)
    writer.add_scalar('Train/Acc', train_acc, epoch)
    writer.add_scalar('Test/loss', test_loss, epoch)
    writer.add_scalar('Test/Acc', test_acc, epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))