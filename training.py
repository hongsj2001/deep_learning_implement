import numpy as np
import matplotlib.pyplot as plt
from model import *
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from optparse import OptionParser


#TODO : Json 파일로 loss와 Weight을 저장

# Device configuration
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(data_dir, batch_size, transform , random_seed, valid_size=0.25, shuffle=True):
    '''
    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    '''

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

#옵션 선택 (제작중)

parser = OptionParser()
parser.add_option("-s", "--seed", default=1, help="the random seed", action="store", type="int", dest="seed")
parser.add_option("-a", "--aug", default='True', help="the random seed", action="store", type="str", dest="aug")
#parser.add_option("-l", "--load", default=False, help="Load weight", action="store", type="str", dest="load")
parser.add_option("-d", "--data", default="Cifar10", help="Select dataset : Cifar10", action="store", type="string", dest="dataset")
parser.add_option("-m", "--model", default="AlexNetCifar10", help="Select models : AlexNet, VGG", action="store", type="string", dest="model")
(options, args) = parser.parse_args()


print('ARG Model', options.model)
print('ARG Augment', options.aug)

normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

if options.aug == 'True':
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



num_epochs = 20
batch_size = 256
learning_rate = 0.005



if options.model == 'AlexNetCifar10':
    model = AlexNetCifar10(num_classes).to(device)
    modelName = 'AlexNetCifar10'
elif options.model == 'VggNetCifar10':
    model = VggNetCifar10(num_classes).to(device)
    modelName = 'VggNetCifar10'
elif options.model == 'VggNet':
    model = VggNet(num_classes).to(device)
    modelName = 'VggNet'






# CIFAR10 dataset 
train_loader, valid_loader, test_loader = get_data(data_dir = './data',batch_size = batch_size, transform=transform,random_seed = 1)


train_acc = []

#모델 불러오기

'''
PATH = 'c:/Users/hong/workspace/deep_learnig_from_scratch/model_weights.pth'
model = AlexNet(num_classes).to(device)
model.load_state_dict(torch.load(PATH))
'''


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
    
    checkpoint_path = f'./checkpoint/{modelName}/checkpoint_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch + 1}')

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