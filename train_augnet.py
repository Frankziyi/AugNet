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
from utils.Dataset import Dataset
from utils.model import ft_fcnet
from utils.utils import set_seed
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser()

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

data_transform_resize = = transforms.Compose([
    transforms.Resize((args.img_bi_h, args.img_bi_w)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}

#image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

image_datasets['train'] = Dataset(image_dir, data_transform, data_transform_resize)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)

dataset_sizes = len(image_datasets['train'])

model = ft_fcnet()
model2 = ft_fcnet()

optimizer_ft = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

#model = DataParallel(model)
model = model.cuda()
model2 = model2.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.state_dict(), save_path)


def train_model(model, optimizer, scheduler, num_epochs):

    scheduler.step()
    model.train()
    model2.train()
    
    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 20)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            inputs, inputs_resize, _, _ = data
            inputs = Variable(inputs.float()).cuda()
            inputs_resize = Variable(inputs_resize.float()).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_resize = model2(inputs_resize)
            #pdb.set_trace()
            #loss compute
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            #pdb.set_trace()

        epoch_loss = running_loss / len(dataloaders)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f}'.format(epoch, epoch_loss))
        #save_network(model, 'test')

	if (epoch + 1) % 20 == 0:
            save_network(model, epoch)

    save_network(model, 'last')

model = train_model(model, optimizer_ft, exp_lr_scheduler, args.num_epochs)
