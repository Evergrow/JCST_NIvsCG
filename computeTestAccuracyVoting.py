# Computing average accuracy on cropped patch (96 x 96) and full-sized image after voting
# This file can also be modified for other patch sizes, i.e., 180 x 180, 120 x 120, etc.

import numpy as np
from PIL import Image
import os
import torch
import glob
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from model_conv1 import ScNet


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

caffe_root = '/.../'
data_root = '.../'

project_root = '/...'
kPrcgNum = 1600

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
        transforms.TenCrop(96),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])


# instantiate model and initialize weights
model = ScNet()
model.cuda()
checkpoint = torch.load(project_root + '/checkpoint_1200.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

imageTmp = []
testTmp = []

testImageDir = caffe_root + data_root + 'NI'
testImageFile = list(glob.glob(testImageDir + '/*.jpg')) + list(glob.glob(testImageDir + '/*.png'))
testImageDir = caffe_root + data_root + 'CG'
testImageFile += list(glob.glob(testImageDir + '/*.jpg')) + list(glob.glob(testImageDir + '/*.png'))
for line in testImageFile:
    image_path = line
    lists = image_path.split('/')
    if lists[-2] == 'NI':
        testClass = 1
    else:
        testClass = 0

    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        test_input = transform(img)
        test_input = test_input.cuda()
        input_var = Variable(test_input, volatile=True)

        ncrops, c, h, w = input_var.size()
        # compute output
        output = model(input_var.view(-1, c, h, w))
        # _, pred = torch.max(output, 1)
        pred = F.softmax(output, dim=1)
        mean = torch.mean(pred, dim=0)
        label = 0
        if mean[1] > 0.5:
            label = 1
        testTmp.append(int(label))  # the predicted label
        imageTmp.append(testClass)

print('The number of full-sized testing images is:', len(imageTmp))


imageLabelNp = np.array(imageTmp)
testLabelNp = np.array(testTmp)

#  Computing average accuracy on patches
result = np.array(imageLabelNp) == np.array(testLabelNp)

prcg_result = result[kPrcgNum:]
google_result = result[:kPrcgNum]
print('The number of patches:', kPrcgNum*2, len(prcg_result), len(google_result))
print('The average accuracy on patches:')
print('The google (NI) accuracy is:', google_result.sum()*1.0/len(google_result))
print('The prcg (CG) accuracy is:', prcg_result.sum()*1.0/len(prcg_result))
print('CG patches misclassified as natural patches (CGmcNI) is:', (len(prcg_result) - prcg_result.sum())*1.0/len(prcg_result))
print('natural patches misclassified as CG patches (NImcCG) is:', (len(google_result) - google_result.sum())*1.0/len(google_result))
print('The average accuracy is:', (google_result.sum()*100.0/len(google_result) + prcg_result.sum()*100.0/len(prcg_result))/ 2)
