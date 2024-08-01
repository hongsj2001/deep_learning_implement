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
from datetime import datetime

#TODO : Json 파일로 loss와 Weight을 저장

#폴더 생성하는 함수
def makedirs(path):
    if not os.path.exists(path): #해당 폴더가 존재하지 않을 경우만 생성
        os.makedirs(path)

#데이터를 불러오는 함수 (수정필요)
def get_data(dataset, data_dir, batch_size, transform , random_seed, valid_size=0.25, shuffle=True):
    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    
    test_loader = torch.utils.data.DataLoader(
        dataset,shuffle=shuffle
    )
    return (train_loader, valid_loader, test_loader)


# Device configuration
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#옵션 선택
parser = OptionParser()
parser.add_option("-s", "--seed", default=1, help="The random seed", action="store", type="int", dest="seed") #Define random seed (Defalut : 1)
parser.add_option("-a", "--aug", default='False', help="Data augmentation", action="store", type="str", dest="aug") #Data Augmentation (Defalut : False)
parser.add_option("-d", "--data", default="Cifar10", help="Select dataset : Cifar10", action="store", type="string", dest="dataset") #Dataset (Defalut : Cifar 10)
parser.add_option("-m", "--model", default="AlexNet", help="Select models : AlexNet, VGG, AlexNetCifar10, VGGCifar10", action="store", type="string", dest="model") #Model (Defalut : AlexNet)
parser.add_option("-e", "--epoch", default=20, help="Epoch", action="store", type="int", dest="epoch") #Epoch (defalut : 20)
parser.add_option("-b", "--batch", default=256, help="Batch size", action="store", type="int", dest="batch") #Batch Size (Defalut : 256)
parser.add_option("-r", "--learningrate", default=0.005, help="Learning rate", action="store", type="float", dest="learningRate") #Learning Rate (defalut : 0.005)
parser.add_option("-t", "--transform", default='False', help="Transform data size (defalut : False) : if you choose this option, cifar 10 dataset image size will be 227x227", action="store", type="string", dest="transform") #Transform data size (defalut : False)
parser.add_option("-l", "--load", default='False', help="Load model", action="store", type="string", dest="load") #Load Data (defalut : False)
parser.add_option("-o", "--optimizer", default='SGD', help="select optimizer : SGD, Momentum, RMSProp, Adagrad, Adam ... (Defalut : SGD)", action="store", type="string", dest="load") #Load Data (defalut : False)
(options, args) = parser.parse_args()


print('ARG seed', options.seed)
print('ARG aug', options.aug)
print('ARG data', options.data)
print('ARG model', options.model)
print('ARG epoch', options.epoch)
print('ARG batch size', options.batch)
print('ARG learning rate', options.learningrate)
print('ARG transform', options.transform)

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    PCANoisePIL(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
if options.transform == 'True':
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    num_classes = 10
else:
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    num_classes = 10

num_epochs = options.epoch
batch_size = options.batch
learning_rate = options.learningrate

#모델 선택
if options.model == 'AlexNet':
    model = AlexNetCifar10(num_classes).to(device)
    modelName = 'AlexNet'
elif options.model == 'AlexNetCifar10':
    model = AlexNetCifar10(num_classes).to(device)
    modelName = 'AlexNetCifar10'
elif options.model == 'VggNet':
    model = VggNet(num_classes).to(device)
    modelName = 'VggNet'
elif options.model == 'VggNetCifar10':
    model = VggNetCifar10(num_classes).to(device)
    modelName = 'VggNetCifar10'

#현재 시간
time = datetime.now()
savefolder = f'./checkpoint/{modelName}/{time}'
makedirs(savefolder)

# CIFAR10 dataset 
train_loader, valid_loader, test_loader = get_data(data_dir = './data',batch_size = batch_size, transform=transform,random_seed = 1)


#모델 불러오기
PATH = 'c:/Users/hong/workspace/deep_learnig_from_scratch/model_weights.pth'
model = VggNetCifar10(num_classes).to(device)
model.load_state_dict(torch.load(PATH))


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)


# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):  
        
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print ('Model1: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    

            del images, labels, outputs
    
        print('Accuracy of the Model1 on the {} validation images: {} %'.format(5000, 100 * correct / total))
        train_acc.append(100 * correct / total)
        print(train_acc)
    
    checkpoint_path = savefolder + f'/checkpoint_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch + 1}')

# Test
with torch.no_grad():
    model.eval()
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
'''
for epoch in range(num_epochs):
    # Test
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
    
        print('Accuracy of the Model on the {} test images: {} %'.format(5000, 100 * correct / total))
        test_acc.append(100 * correct / total)
        print(test_acc)
'''
#torch.save(model.state_dict(), PATH)



'''
#모델 성능이 괜찮다면 추후에 다시 사용하기 위해서 모델을 저장할 수 있음 
# odels.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')


model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''


x = np.arange(num_epochs)
plt.plot(x, train_acc, marker="x", markevery=100, label="train")
plt.xlabel("epoch")
plt.ylabel("acurracy(%)")
plt.ylim(0, 100)
plt.legend()
plt.show()