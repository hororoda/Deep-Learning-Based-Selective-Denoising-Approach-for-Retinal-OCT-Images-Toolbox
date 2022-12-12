"""
train models to classify retinal OCT images into NRR and NF images
"""


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import random


# torch base dir
os.environ['TORCH_HOME'] = ''
cudnn.benchmark = False

'''
parameters
'''
# model name
model_name = ''
# target class num
num_classes = 2
# batch size
batch_size = 32
# epoch num
num_epochs = 25
# finetuning or feature_exact
feature_extract = False
# dataset path
data_dir = ''
# model save path
model_save_path = ''
# tensorboard path
tensorboard_save_path = ''


def train_model(model, dataloaders, criterion, optimizer, scheduler, model_save_name, num_epochs=25,
                is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # accuracy and loss curve save path
    apart_train_accu_log_dir = tensorboard_save_path + '/apart/accu/train'
    apart_train_loss_log_dir = tensorboard_save_path + '/apart/loss/train'
    apart_val_accu_log_dir = tensorboard_save_path + '/apart/accu/val'
    apart_val_loss_log_dir = tensorboard_save_path + '/apart/loss/val'

    together_train_accu_log_dir = tensorboard_save_path + '/together/accu/train'
    together_train_loss_log_dir = tensorboard_save_path + '/together/loss/train'
    together_val_accu_log_dir = tensorboard_save_path + '/together/accu/val'
    together_val_loss_log_dir = tensorboard_save_path + '/together/loss/val'

    apart_train_accu_writer = SummaryWriter(log_dir=apart_train_accu_log_dir)
    apart_train_loss_writer = SummaryWriter(log_dir=apart_train_loss_log_dir)
    apart_val_accu_writer = SummaryWriter(log_dir=apart_val_accu_log_dir)
    apart_val_loss_writer = SummaryWriter(log_dir=apart_val_loss_log_dir)

    together_train_accu_writer = SummaryWriter(log_dir=together_train_accu_log_dir)
    together_train_loss_writer = SummaryWriter(log_dir=together_train_loss_log_dir)
    together_val_accu_writer = SummaryWriter(log_dir=together_val_accu_log_dir)
    together_val_loss_writer = SummaryWriter(log_dir=together_val_loss_log_dir)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # output time cost for each epoch
        epoch_since = time.time()

        # training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # training mode
            else:
                model.eval()  # evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward propagation (track history only in training)
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward propagation (optimize only in training)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                apart_train_accu_writer.add_scalar('accu/train', epoch_acc, epoch)
                apart_train_loss_writer.add_scalar('loss/train', epoch_loss, epoch)
                together_train_accu_writer.add_scalar('accu', epoch_acc, epoch)
                together_train_loss_writer.add_scalar('loss', epoch_loss, epoch)

            else:
                apart_val_accu_writer.add_scalar('accu/val', epoch_acc, epoch)
                apart_val_loss_writer.add_scalar('loss/val', epoch_loss, epoch)
                together_val_accu_writer.add_scalar('accu', epoch_acc, epoch)
                together_val_loss_writer.add_scalar('loss', epoch_loss, epoch)

            epoch_time_elapsed = time.time() - epoch_since
            print('This epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # model save
                torch.save(model, model_save_path + '/model/' + model_save_name + '.pth')  # model
                torch.save(best_model_wts, model_save_path + '/weights/best_model_wts.pth')  # weights
                torch.save(optimizer.state_dict(), model_save_path + '/optimizer/optimizer_state_dict.pth')  # optimizer

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# model initialization
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "":
        # model
        model_ft =
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input size
        input_size = 224

        # for example :
        # if model_name == "":
        #     Densenet161
        #     model_ft = models.densenet161(pretrained=use_pretrained)
        #     set_parameter_requires_grad(model_ft, feature_extract)
        #     num_ftrs = model_ft.classifier.in_features
        #     model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        #     input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# configure the model
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model
print(model_ft)

'''
Data loading and transformation
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                    for x in ['train', 'val']}

# Detect
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Create the Optimizer
'''
model_ft = model_ft.to(device)

'''
finetuning: updating all parameters. feature_extract == false.
feature extract: updating the parameters just initialized. feature_extract == true.
'''
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# create pptimizer
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, nesterov=True)

'''
Setup and Run
'''
# loss founction
criterion = nn.CrossEntropyLoss()
# scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, model_name,
                             num_epochs=num_epochs,
                             is_inception=(model_name == 'inception'))
