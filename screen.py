"""
classify retinal OCT images into NRR and NF images using the trained model
"""


import torch
from torchvision import transforms
import shutil
import MyImageFolder
import time


'''
parameters
'''
# model dir
model_dir = ''
# images that need to be screend
data_dir = ''
# NRR img save dir
NRR_img_save_path = ''
# NF img save dir
NF_img_save_path = ''
# input size
input_size = 224
# batch size
batch_size =


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_dir)
model = model.to(device)
model.eval()


# load data
data_transform = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = MyImageFolder.FilterableImageFolder(data_dir, data_transform, valid_classes=['img'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def getfilename(str1):
    for i in range(1, len(str1)):
        if str1[len(str1) - i] == '/'or str1[len(str1) - i] == '\\':
            idx = len(str1) - i + 1
            break
    return str1[idx:]


with torch.no_grad():
    for i, (batch_images, _) in enumerate(dataloader, 0):
        if torch.cuda.is_available():
            batch_images = batch_images.cuda()
        out = model(batch_images)
        prediction = torch.max(out, 1)[1]

        filepath, _ = dataloader.dataset.samples[i]
        if prediction[0] == 0:
            # copy NF images to tagert dir
            shutil.copy(filepath, NF_img_save_path + '/' + getfilename(filepath))
        else:
            # copy NRR images to tagert dir
            shutil.copy(filepath, NRR_img_save_path + '/' + getfilename(filepath))

