import numpy as np
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models

# Device configuration
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
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

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

#옵션 선택 (제작중)
'''
parser = OptionParser()
parser.add_option("-s", "--seed", default=0, help="the random seed", action="store", type="int", dest="seed")
parser.add_option("-ㅣ", "--load", default=False, help="Load Data", action="store", type="bool", dest="load")

(options, args) = parser.parse_args()
'''


# CIFAR10 dataset 
train_loader, valid_loader = get_train_valid_loader(data_dir = './data',batch_size = 64, augment = False, random_seed = 1)

test_loader = get_test_loader(data_dir = './data', batch_size = 64)

num_classes = 10
num_epochs = 50
batch_size = 128
learning_rate = 0.005

model1 = AlexNet(num_classes).to(device)

train_acc1 = []

#모델 불러오기

'''
PATH = 'c:/Users/hong/workspace/deep_learnig_from_scratch/model_weights.pth'
model = AlexNet(num_classes).to(device)
model.load_state_dict(torch.load(PATH))
'''


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs1 = model1(images)
        loss1 = criterion(outputs1, labels)
        
        # Backward and optimize
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()


    print ('Model1: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))
            
    # Validation
    with torch.no_grad():
        correct1 = 0
        correct2 = 0
        total1 = 0
        total2 = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs1 = model1(images)
            _, predicted1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted1 == labels).sum().item()

    

            del images, labels, outputs1
    
        print('Accuracy of the Model1 on the {} validation images: {} %'.format(5000, 100 * correct1 / total1))
        train_acc1.append(100 * correct1 / total1)
        print(train_acc1)

#torch.save(model.state_dict(), PATH)



'''
#모델 성능이 괜찮다면 추후에 다시 사용하기 위해서 모델을 저장할 수 있음 
# odels.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')


model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''

x = np.arange(num_epochs)
plt.plot(x, train_acc1, marker="x", markevery=100, label="Relu")
plt.xlabel("epoch")
plt.ylabel("acurracy(%)")
plt.ylim(0, 100)
plt.legend()
plt.show()