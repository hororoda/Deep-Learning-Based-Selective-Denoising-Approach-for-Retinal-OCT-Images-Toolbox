"""
calculate and plot confusion matrix
"""


from __future__ import print_function
from __future__ import division
import fontTools.cffLib
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn


'''
parameters
'''
cudnn.benchmark = False
# model dir
model_dir = ""
# test set
test_data_dir = ''
# classes
target_names = ['NF', 'NRR']
# input size
input_size = 224
# batch size
batch_size =


def plot_confusion_matrix(cm, target_names, cmap=None, normalize=True):
    # format
    num = dict(fontsize=24,
               color='black',
               family='Times New Roman',
               weight='light',
               )
    label = dict(fontsize=24,
                 color='black',
                 family='Times New Roman',
                 weight='light',
                 # style='italic',
                 )
    tick = dict(fontsize=24,
                color='black',
                family='Times New Roman',
                weight='light',
                )

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontdict=tick)
        plt.yticks(tick_marks, target_names, fontdict=tick)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontdict=num)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontdict=num)

    plt.tight_layout()

    plt.ylabel('True label', fontdict=label)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontdict=label)
    plt.show()


model = torch.load(model_dir)

# load data
data_transform = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = torchvision.datasets.ImageFolder(test_data_dir, data_transform)
test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

true = torch.empty(1)
pred = torch.empty(1)

true = true.to(device)
pred = pred.to(device)

for batch_images, batch_labels in test_dataloader:
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

    out = model(batch_images)

    prediction = torch.max(out, 1)[1]

    true = torch.cat((true, batch_labels), dim=0)
    pred = torch.cat((pred, prediction), dim=0)

true = true[1:]
pred = pred[1:]

true = true.cpu().numpy()
pred = pred.cpu().numpy()

cm = confusion_matrix(true, pred)

plot_confusion_matrix(cm, target_names, cmap=None, normalize=False)
