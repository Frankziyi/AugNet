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
from utils.Dataset import Dataset, DatasetTri, DatasetTriphard
from utils.model import ft_net, ft_fcnet
from utils.triphard import UnsupervisedTriphard
from utils.random_erasing import RandomErasing
from torch.nn.parallel import DataParallel
from utils.resnet import remove_fc
import utils.my_transforms as my_transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr_decay_epochs', type=int, default=30)
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
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    my_transforms.RandomCrop(range=(0.70, 0.95)),
    #transforms.RandomCrop(size=(384, 128)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    #RandomErasing(probability = 1.0, mean=[0.0, 0.0, 0.0])
    ])

image_datasets = {}

image_datasets['train'] = DatasetTri(image_dir, data_transform, data_transform2)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)

dataset_sizes = len(image_datasets['train'])

triplet_loss = nn.TripletMarginLoss(margin=3.0)

model_structure = ft_fcnet()

def load_network(network):
    save_path = os.path.join(args.pretrained_path, 'pretrained_weight.pth')
    network.load_state_dict({'model.' + k : v for k,v in remove_fc(torch.load(save_path)).items()}, strict = False)
    return network

model = load_network(model_structure)
model2 = load_network(model_structure)

optimizer_ft = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9, weight_decay=5e-4)
optimizer_ft2 = optim.SGD(model2.parameters(), lr = 0.0001, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)
exp_lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer_ft2, step_size=args.lr_decay_epochs, gamma=0.1)

model = DataParallel(model)
model = model.cuda()
model2 = DataParallel(model2)
model2 = model2.cuda()

def save_network(network1, network2, epoch_label):
    save_filename1 = 'net1_%s.pth'% epoch_label
    save_path1 = os.path.join(args.model_save_dir, save_filename1)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network1.state_dict(), save_path1)
    
    save_filename2 = 'net2_%s.pth'% epoch_label
    save_path2 = os.path.join(args.model_save_dir, save_filename2)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network2.state_dict(), save_path2)


def train_model(model, optimizer, scheduler, optimizer2, scheduler2, num_epochs):
    
    scheduler.step()
    model.train()
    model2.train()

    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 20)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            
            inputs, pos, neg, _, _ = data
            #inputs, pos, _, _ = data
            inputs = Variable(inputs.float()).cuda()
            pos = Variable(pos.float()).cuda()
            neg = Variable(neg.float()).cuda()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            features = model(inputs)
            features_pos = model2(pos)
            features_neg = model(neg)
            #features_resize_neg = model2(neg)
            #pdb.set_trace()
            #loss = triplet_loss(features, features_pos, features_neg) + triplet_loss(features, features_pos, feature_resize_neg)
            loss = triplet_loss(features, features_pos, features_neg)
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss += loss.data.item()
            #pdb.set_trace()

        epoch_loss = running_loss / len(dataloaders)
        epoch_acc = running_corrects * 1.0 / float(dataset_sizes)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f}'.format(epoch, epoch_loss))

	if (epoch + 1) % 5 == 0:
            save_network(model, model2, epoch)

    #save_network(model, 'last')
    
model = train_model(model, optimizer_ft, exp_lr_scheduler, optimizer_ft2, exp_lr_scheduler2, args.num_epochs)
