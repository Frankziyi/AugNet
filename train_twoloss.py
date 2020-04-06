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
from utils.Dataset import DatasetAug, DatasetTri, DatasetTriphard
from utils.model import ft_net, ft_fcnet
from utils.triphard import UnsupervisedTriphard
from utils.random_erasing import RandomErasing
from torch.nn.parallel import DataParallel
from utils.resnet import remove_fc
import utils.my_transforms as my_transforms
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr_decay_epochs', type=int, default=50)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=128)
parser.add_argument('--img_bi_h', type=int, default=512)
parser.add_argument('--img_bi_w', type=int, default=256)
parser.add_argument('--img_tri_h', type=int, default=512)
parser.add_argument('--img_tri_w', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--l2', type=float, default=1.0)

args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

data_transform2 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    my_transforms.RandomCrop(range=(0.70, 0.95)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((args.img_bi_h, args.img_bi_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    #RandomErasing(probability = 1.0, mean=[0.0, 0.0, 0.0])
    ])


data_transform3 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    my_transforms.RandomCrop(range=(0.70, 0.95)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((args.img_tri_h, args.img_tri_w)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}

#image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

image_datasets['train'] = DatasetAug(image_dir, data_transform, data_transform2, data_transform3)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)

dataset_sizes = len(image_datasets['train'])

#triplet_loss = nn.TripletMarginLoss(margin=0.3)

triplet_loss = UnsupervisedTriphard(margin=0.3)

model_structure = ft_fcnet()

def load_network(network):
    save_path = os.path.join(args.pretrained_path, 'pretrained_weight.pth')
    network.load_state_dict({'model.' + k : v for k,v in remove_fc(torch.load(save_path)).items()}, strict = False)
    return network

model = load_network(model_structure)

#optimizer_ft = optim.SGD(model.parameters(), lr = 0.0, momentum=0.0, weight_decay=0)
optimizer_ft = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9, weight_decay=5e-4)

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
        l2 = 0.0

        for data in dataloaders:
            
            inputs, pos, reszinputs, _, _, _ = data
            #inputs, pos, _, _ = data
            inputs = Variable(inputs.float()).cuda()
            pos = Variable(pos.float()).cuda()
            reszinputs = Variable(reszinputs.float()).cuda()
            optimizer.zero_grad()
            features = model(inputs)
            features_pos = model(pos)
            features_resz = model(reszinputs)
            #features_neg = model(neg)
            #loss = triplet_loss(features, features_pos, features_neg)
            loss = triplet_loss(features, features_pos)
            l2loss = args.l2 * F.pairwise_distance(features, features_resz, p=2).sum()
            #pdb.set_trace()
            loss.backward(retain_graph=True)
            l2loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            l2 += l2loss.data.item()

        epoch_loss = running_loss / len(dataloaders)
        epoch_l2_loss = l2 / len(dataloaders)
        epoch_acc = running_corrects * 1.0 / float(dataset_sizes)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f}  L2: {:.4f}'.format(epoch, epoch_loss, epoch_l2_loss))

	if (epoch + 1) % 5 == 0:
            save_network(model, epoch)

    save_network(model, 'last')
    
model = train_model(model, optimizer_ft, exp_lr_scheduler, args.num_epochs)
