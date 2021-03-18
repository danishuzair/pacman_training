import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class PacmanImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1, sample2 = self.loader(path)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        sample = torch.stack((sample1, sample2),dim=1).squeeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def is_valid_file(path):
    if path.find("p.jpg") != -1:
        return False
    else:
        return True

def loader(path):
    with open(path, 'rb') as f:
        img1 = Image.open(f)
        img1c = img1.convert()
    pathsplit = path.split(".")
    pathnew = pathsplit[0] + 'p.' + pathsplit[1]
    with open(pathnew, 'rb') as f:
        img2 = Image.open(f)
        img2c = img2.convert()
    return img1c, img2c

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad((64,0),fill=0)
    # transforms.Normalize(mean=0, std=255),
])

data_dir = '/home/danish/Desktop/CodingProjects/Data/Pacman_Data/PacmanData_BW_Proof'
image_datasets = PacmanImageFolder(os.path.join(data_dir),
    transform=transform,
    is_valid_file=is_valid_file,
    loader=loader)
print(image_datasets.class_to_idx)

batch_size = 4
shuffle_dataset = True
random_seed = 42
validation_split = 0.05

dataset_size = len(image_datasets)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
dataset_sizes = {}
dataset_sizes['train'] = len(train_indices)
dataset_sizes['val'] = len(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, 
                                           sampler=train_sampler, num_workers=4)
# for step, (x,y) in enumerate(train_loader):
#     print(x.shape)

validation_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=4)
# for step, (x,y) in enumerate(validation_loader):
#     print(x.shape)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'train':
                dataloaders = train_loader
            else:
                dataloaders = validation_loader

            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

screen_height = 640
screen_width = 640
number_actions = 4
model_ft = DQN(screen_height,screen_width,number_actions).to(device)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


## Data without any training / validating splits
# dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True,  num_workers=4)
# for step, (x, y) in enumerate(dataloader):
#     print(x.shape)

## DISPLAYING THE DATA
# plt.gray()
# dataiter = iter(dataloader)
# images,labels = dataiter.next()
# plt.imshow(images[0][0])
# plt.show(block=True)
# plt.imshow(images[0][0].cpu().detach().numpy())
# plt.show(block=True)
# plt.imshow(images[0][1].cpu().detach().numpy())
# plt.show(block=True)

# for idx, (sample, target) in enumerate(image_datasets):
#     print(sample.shape)
#     print(sample, target)
# print(image_datasets.samples)
# print(image_datasets.targets)

# generator = iter(dataloader)
# max_steps = 10
# for i in range(max_steps):
#     try:
#         # Samples the batch
#         x, y = next(generator)
#     except StopIteration:
#         # restart the generator if the previous generator is exhausted.
#         generator = iter(trainloader)
#         x, y = next(generator)

# print(dir(dataloader))
# it = iter(dataloader)
# first = next(it)
# print(dir(first))
# print(first.dataset)
# second = next(it)

# print(image_datasets.imgs)
# print(image_datasets.classes)
# print(image_datasets.imgs[0])
# print(dir(dataloader))
# inputs, classes = next(iter(dataloader))


# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch)
#     print(sample_batched)
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['landmarks'].size())