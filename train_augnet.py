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
from utils.Dataset import Dataset, DatasetTri
from utils.model import ft_fcnet
from utils.resnet import remove_fc
from utils.triphard import UnsupervisedTriphard
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

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
parser.add_argument('--img_bi_h', type=int, default=512)
parser.add_argument('--img_bi_w', type=int, default=256)
parser.add_argument('--gaussian_blur', type=bool, default=True)
parser.add_argument('--salt_and_pepper_noise', type=bool, default=True)
parser.add_argument('--random_crop', type=bool, default=True)
parser.add_argument('--random_erasing', type=bool, default=True)

args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

data_transform_resize = transforms.Compose([
    transforms.Resize((args.img_bi_h, args.img_bi_w)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomCrop(size=(384,128)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}

#image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

image_datasets['train'] = DatasetTri(image_dir, data_transform, data_transform_resize)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)

dataset_sizes = len(image_datasets['train'])

triplet_loss = UnsupervisedTriphard(margin=2.0)

triplet_loss = nn.TripletMarginLoss(margin=2.0)

model = ft_fcnet()
model2 = ft_fcnet()

def load_network(network):
    save_path = os.path.join(args.pretrained_path, 'pretrained_weight.pth')
    network.load_state_dict({'model.'+k:v for k, v in remove_fc(torch.load(save_path)).items()}, strict=False)
    return network

model = load_network(model)
model2 = load_network(model2)

optimizer_ft = optim.SGD(model.parameters(), lr = 0.00002, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

optimizer_ft2 = optim.SGD(model2.parameters(), lr = 0.00002, momentum = 0.9, weight_decay = 5e-4)

exp_lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer_ft2, step_size=args.lr_decay_epochs, gamma=0.1)


model = DataParallel(model)
model2 = DataParallel(model2)
model = model.cuda()
model2 = model2.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.state_dict(), save_path)


def train_model(model, optimizer, scheduler, optimizer2, scheduler2, num_epochs):

    scheduler.step()
    scheduler2.step()
    model.train()
    model2.train()
    
    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 20)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            inputs, inputs_resize, neg,  _, _ = data
            inputs = Variable(inputs.float()).cuda()
            inputs_resize = Variable(inputs_resize.float()).cuda()
            neg = Variable(neg.float()).cuda()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = model(inputs)
            outputs_resize = model2(inputs_resize)
            outputs_neg = model(neg)
            #loss compute
            #loss = triplet_loss(outputs, outputs_resize)
            loss = triplet_loss(outputs, outputs_resize, outputs_neg)
            #pdb.set_trace()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                optimizer2.step()
            running_loss += loss.data.item()
            #pdb.set_trace()

        epoch_loss = running_loss / len(dataloaders)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f}'.format(epoch, epoch_loss))
        #save_network(model, 'test')

	if (epoch + 1) % 10 == 0:
            save_network(model, epoch)

    save_network(model, 'last')

model = train_model(model, optimizer_ft, exp_lr_scheduler, optimizer_ft2, exp_lr_scheduler2, args.num_epochs)
