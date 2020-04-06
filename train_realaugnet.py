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
from utils.Dataset import Dataset, DatasetTri, DatasetAug
from utils.model import ft_fcnet
from utils.resnet import remove_fc
from utils.triphard import UnsupervisedTriphard
from torch.nn.parallel import DataParallel
import utils.my_transforms as my_transforms
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--bilinear_interpolation', type=bool, default=True)
parser.add_argument('--img_bi_h', type=int, default=576)
parser.add_argument('--img_bi_w', type=int, default=192)
parser.add_argument('--img_tri_h', type=int, default=768)
parser.add_argument('--img_tri_w', type=int, default=256)
parser.add_argument('--gaussian_blur', type=bool, default=True)
parser.add_argument('--salt_and_pepper_noise', type=bool, default=True)
parser.add_argument('--random_crop', type=bool, default=True)
parser.add_argument('--random_erasing', type=bool, default=True)
parser.add_argument('--tripletloss', type=float, default=1.0)
parser.add_argument('--l2loss', type=float, default=1.0)


args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

data_transform_resize = transforms.Compose([
    #transforms.Resize((args.img_bi_h, args.img_bi_w)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    #transforms.RandomCrop(size=(384,128)),
    my_transforms.RandomCrop(range=(0.70,0.95)),
    transforms.Resize((args.img_bi_h, args.img_bi_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

data_transform_resize2 = transforms.Compose([
    #transforms.Resize((args.img_tri_h, args.img_tri_w)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    #transforms.RandomCrop(size=(576,192)),
    my_transforms.RandomCrop(range=(0.70,0.95)),
    transforms.Resize((args.img_tri_h, args.img_tri_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}

#image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

image_datasets['train'] = DatasetAug(image_dir, data_transform, data_transform_resize, data_transform_resize2)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)

dataset_sizes = len(image_datasets['train'])

triplet_loss = UnsupervisedTriphard(margin=0.3)

#triplet_loss = nn.TripletMarginLoss(margin=12.0)

model = ft_fcnet()
model2 = ft_fcnet()
#model3 = ft_fcnet()

def load_network(network):
    save_path = os.path.join(args.pretrained_path, 'pretrained_weight.pth')
    network.load_state_dict({'model.'+k:v for k, v in remove_fc(torch.load(save_path)).items()}, strict=False)
    return network

model = load_network(model)
model2 = load_network(model2)
#model3 = load_network(model3)

optimizer_ft = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

optimizer_ft2 = optim.SGD(model2.parameters(), lr = 0.0001, momentum = 0.9, weight_decay = 5e-4)

exp_lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer_ft2, step_size=args.lr_decay_epochs, gamma=0.1)

#optimizer_ft3 = optim.SGD(model3.parameters(), lr = 0.0001, momentum = 0.9, weight_decay = 5e-4)

#exp_lr_scheduler3 = optim.lr_scheduler.StepLR(optimizer_ft3, step_size=args.lr_decay_epochs, gamma=0.1)

model = DataParallel(model)
model2 = DataParallel(model2)
#model3 = DataParallel(model3)
model = model.cuda()
model2 = model2.cuda()
#model3 = model3.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.state_dict(), save_path)

def save_network2(network1, network2, epoch_label):
    save_filename1 = 'net1_%s.pth'% epoch_label
    save_path1 = os.path.join(args.model_save_dir, save_filename1)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network1.state_dict(), save_path1)

    save_filename2 = 'net2_%s.pth'% epoch_label
    save_path2 = os.path.join(args.model_save_dir, save_filename2)
    torch.save(network2.state_dict(), save_path2)

def train_model(model, optimizer, scheduler, optimizer2, scheduler2, num_epochs):

    scheduler.step()
    scheduler2.step()
    #scheduler3.step()
    model.train()
    model2.train()
    #model3.train()
    
    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 20)

        running_loss1 = 0.0
        running_loss2 = 0.0
        l2_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            inputs, inputs_resize, inputs_resize2, neg, _, _ = data
            inputs = Variable(inputs.float()).cuda()
            inputs_resize = Variable(inputs_resize.float()).cuda()
            inputs_resize2 = Variable(inputs_resize2.float()).cuda()
            #neg = Variable(neg.float()).cuda()

            optimizer.zero_grad()
            optimizer2.zero_grad()
            anchor1 = model(inputs)
            pos1 = model(inputs_resize)
            #neg1 = model(neg)
            anchor2 = model2(inputs)
            pos2 = model2(inputs_resize2)
            #neg2 = model2(neg)
            #anchor_l2 = F.pairwise_distance(anchor1, anchor2, p=2)
            loss1 = triplet_loss(anchor1, pos1)
            loss2 = triplet_loss(anchor2, pos2)
            l2_distance = F.pairwise_distance(pos1, pos2, p=2).sum()
            aug_l2 = args.l2loss * l2_distance
            #pdb.set_trace()
            running_loss1 += loss1.data.item()
            running_loss2 += loss2.data.item()
            l2_loss += aug_l2.data.item()
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            aug_l2.backward()
            optimizer.step()
            optimizer2.step()
            #pdb.set_trace()

        epoch_loss1 = running_loss1 / len(dataloaders)
        epoch_loss2 = running_loss2 / len(dataloaders)
        epoch_l2_loss = l2_loss / len(dataloaders)
        #pdb.set_trace()
	print ('Epoch:{:d} TripletLoss1: {:.4f}  TripletLoss2: {:.4f}  L2Loss: {:.4f}'.format(epoch, epoch_loss1, epoch_loss2, epoch_l2_loss))
        #save_network(model, 'test')

	if (epoch + 1) % 5 == 0:
            save_network2(model, model2, epoch)

model = train_model(model, optimizer_ft, exp_lr_scheduler, optimizer_ft2, exp_lr_scheduler2, args.num_epochs)