import numpy as np
import matplotlib.pyplot as plt
import os
from model import *
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from optparse import OptionParser

def get_data(data_dir, batch_size, transform, shuffle=True):

    test_dataset = datasets.CIFAR10(root=data_dir, train = False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=shuffle)
    

    return test_loader


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = OptionParser()
parser.add_option("-m", "--model", default="AlexNet", help="Select models : AlexNet, VGG, AlexNetCifar10, VGGCifar10", action="store", type="string", dest="model") #Model (Defalut : AlexNet)
parser.add_option("-l", "--load", default='False', help="Load model", action="store", type="string", dest="load") #Load Data (defalut : False)
(options, args) = parser.parse_args()

normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 데이터셋의 평균값
        std=[0.2023, 0.1994, 0.2010]
    )
transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

train_loader, valid_loader, test_loader, num_classes = get_data(data_dir = './data/Cifar10', transform= transform, batch_size = 256)

if(options.load == 'True'):
    loadPath = str(input('\n enter the data path : '))

if options.model == 'AlexNet':
    model = AlexNet(10).to(device)
    modelName = 'AlexNet'
elif options.model == 'AlexNetCifar10':
    model = AlexNetCifar10(10).to(device)
    modelName = 'AlexNetCifar10'
elif options.model == 'VggNet':
    model = VggNet(10).to(device)
    modelName = 'VggNet'
elif options.model == 'VggNetCifar10':
    model = VggNetCifar10(10).to(device)
    modelName = 'VggNetCifar10'

model.load_state_dict(torch.load(loadPath))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    

        del images, labels, outputs
    
    print('Accuracy of the Model on the test images: {} %'.format(100 * correct / total))


data_dir = './data/Cifar10'
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

# CIFAR-10 데이터셋에 대한 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 training dataset 불러오기
train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

# CIFAR-10 test dataset 불러오기
test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# CIFAR-10 데이터셋으로부터 클래스 레이블 가져오기
class_labels = train_dataset.classes

def imshow(images, labels):
    batch_size = images.shape[0]
    
    fig, axarr = plt.subplots(1, batch_size, figsize=(10, 4))
    
    for i in range(batch_size):
        img = images[i] #/ 2 + 0.5  # 정규화 복원
        npimg = img.numpy()
        axarr[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axarr[i].set_title(class_labels[labels[i].item()])
        axarr[i].axis('off')

    plt.show()

# 학습 데이터의 일부를 가져옴
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
images, labels = next(iter(dataloader))

# 이미지와 레이블을 함께 출력
imshow(images, labels)