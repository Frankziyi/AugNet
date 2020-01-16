import numpy as np
import torch
import os
import argparse
import torchvision
import pdb
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from utils.sampler import RandomIdentitySampler,RandomSampler
from utils.Dataset import Dataset, DatasetTriplet
from utils.model import ft_net, ft_fcnet
from utils.triphard import TripHard
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=128)
parser.add_argument('--img_bi_h', type=int, default=512)
parser.add_argument('--img_bi_w', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

data_transform2 = transforms.Compose([
    transforms.Resize((args.img_bi_h, args.img_bi_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, sampler=RandomIdentitySampler(image_datasets['train'].imgs), num_workers=8)

dataset_sizes = len(image_datasets['train'])

triphard_loss = TripHard(margin=0.3)

model = ft_fcnet()

optimizer_ft = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

model = DataParallel(model)
model = model.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.state_dict(), save_path)


def train_model(model, optimizer, scheduler, num_epochs):
    
    scheduler.step()
    model.train()

    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 20)

        running_loss = 0.0
        running_corrects = 0

        #image_datasets['train'].Shuffle()
        #dataloaders =  torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)

        for data in dataloaders:
            inputs, labels = data
            inputs = Variable(inputs.float()).cuda()
            labels = Variable(labels.float()).cuda()
            optimizer.zero_grad()
            features = model(inputs)
            #pdb.set_trace()
            loss = triphard_loss(features, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            #pdb.set_trace()

        epoch_loss = running_loss / len(dataloaders)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f}'.format(epoch, epoch_loss))

	if (epoch + 1) % 20 == 0:
            save_network(model, epoch)

    save_network(model, 'last')
    
model = train_model(model, optimizer_ft, exp_lr_scheduler, args.num_epochs)
