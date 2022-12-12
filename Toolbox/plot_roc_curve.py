"""
plot ROC curve
"""


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from matplotlib.pyplot import MultipleLocator


def del_tensor_ele_n(arr, index, n):
    arr1 = arr[0:index]
    arr2 = arr[index + n:]
    return torch.cat((arr1, arr2), dim=0)


def test(model_dir, val_data_dir):
    model = torch.load(model_dir)

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = torchvision.datasets.ImageFolder(val_data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    total_out = torch.empty(1, 5)
    total_labels = torch.empty(1)
    total_out = total_out.to(device)
    total_labels = total_labels.to(device)
    for (imgs, labels) in dataloader:
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            total_out = torch.cat((total_out, outputs), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)

    total_out = total_out[1:]
    total_labels = total_labels[1:]

    return total_out, total_labels


# model dir
model_dir = ""
# test set
test_data_dir = ''
# num of classes
num_classes = 5

test_out, test_labels = test(model_dir, test_data_dir)
scores = torch.softmax(test_out, dim=1).cpu().numpy()
binary_label = label_binarize(test_labels.cpu().numpy(), classes=list(range(num_classes)))  # num_classes=4


fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# format
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

for i in range(5):
    plt.plot(fpr[i], tpr[i], lw=2,  # lw:line width
             label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")


# axises
x_major_locator = MultipleLocator(0.2)
y_major_locator = MultipleLocator(0.2)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()

