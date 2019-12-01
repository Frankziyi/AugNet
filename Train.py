import numpy as np
import torch
import os
import argparse
import torchvision
import pdb
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils.sampler import RandomIdentitySampler
from utils.resnet import resnet50
from utils.model import ft_net
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--bilinear_interpolation', type=bool, default=True)
parser.add_argument('--gaussian_blur', type=bool, default=True)
parser.add_argument('--salt_and_pepper_noise', type=bool, default=True)
parser.add_argument('--random_crop', type=bool, default=True)
parser.add_argument('--random_erasing', type=bool, default=True)

args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

#image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, x),data_transform[x])
#                  for x in ['train', 'val']}

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size,
                                            sampler=RandomIdentitySampler(image_datasets['train'].imgs),
                                            num_workers=8)
dataset_sizes = len(image_datasets['train'])

model = ft_net(751)
base_parameters = model.parameters()

optimizer_ft = optim.SGD(base_parameters, lr = 0.1, momentum=0.9)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

loss_function = torch.nn.CrossEntropyLoss()

#model = DataParallel(model)
model = model.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.cpu().state_dict(), save_path)


def train_model(model, optimizer, scheduler, num_epochs):

    #model_weights = model.state_dict()

    #save_network(model, 0)

    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 12)

        #scheduler.step()
        model.cuda()
        #model.train(True)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            inputs, labels = data
            inputs = Variable(inputs.float()).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_corrects += torch.sum(pred == labels.data).item()
            #pdb.set_trace()

        epoch_loss = running_loss
        epoch_acc = running_corrects * 1.0 / float(dataset_sizes)
        #pdb.set_trace()
	print ('Epoch:{:d} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

	#if (epoch + 1) % 20 == 0:
        #    save_network(model, epoch)

    #save_network(model, 'last')

model = train_model(model, optimizer_ft, exp_lr_scheduler, args.num_epochs)
