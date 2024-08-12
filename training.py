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

#TODO : 여러가지 데이터셋을 사용해보자 (ex. 이미지넷에서 클래스를 줄여서 사용 or Kaggle에서 데이터셋들 찾아보기)
#cifar10을 사용했을 때 논문보다 정확도가 낮게 나오는 이유 : 첫 번째 레이어에서 필터 크기가 너무 커서 그럴수도 (논문에서는 Cifar10에 맞춰서 3x3 사용) 
#batch size 바꿔서 비교 : 256, 512, 1024, 2048 ...
#실험돌리기 전에 어떻게 될 지 예상하고 결과와 비교
#ResNet에서 Residual을 쓸 때랑 쓰지 않을 때 차이 확인
#여러가지 하이퍼 파라미터와 바꿀 수 있는 요소들 바꿔보고 비교 및 대조표 작성 (Optimizer, Learning Rate, Batch size ...)
# + Data Augmentation 구현
#데이터 임의로 불러와서 결과 출력

#print and save log
def printf(inputs, file):
    print(inputs)
    print(inputs, file= file)

# make directory
def makedirs(path):
    if not os.path.exists(path): #해당 폴더가 존재하지 않을 경우만 생성
        os.makedirs(path)

#load data
def get_data(dataset, data_dir, batch_size, transform, random_seed, valid_size=0.25, shuffle=True):

    if(dataset == 'MNIST'):
        train_dataset = datasets.mnist(root=data_dir, train = True, download = True, transform = transform)
        valid_dataset = datasets.mnist(root=data_dir, train = True, download = True, transform = transform)
        test_dataset = datasets.mnist(root=data_dir, train = False, download = True, transform = transform)
        num_classes = 10
    else:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train = False, download=True, transform=transform)
        num_classes = 10

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=shuffle)
    

    return (train_loader, valid_loader, test_loader, num_classes)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#select options
parser = OptionParser()
parser.add_option("-s", "--seed", default=1, help="The random seed", action="store", type="int", dest="seed") #Define random seed (Defalut : 1)
#parser.add_option("-a", "--aug", default='False', help="Data augmentation", action="store", type="str", dest="aug") #Data Augmentation (Defalut : False)
parser.add_option("-d", "--dataset", default="Cifar10", help="Select dataset : Cifar10, MNIST ...", action="store", type="string", dest="dataset") #Dataset (Defalut : Cifar 10)
parser.add_option("-m", "--model", default="AlexNet", help="Select models : AlexNet, VGG, AlexNetCifar10, VGGCifar10", action="store", type="string", dest="model") #Model (Defalut : AlexNet)
parser.add_option("-e", "--epoch", default=25, help="Epoch", action="store", type="int", dest="epoch") #Epoch (defalut : 20)
parser.add_option("-b", "--batch", default=64, help="Batch size", action="store", type="int", dest="batch") #Batch Size (Defalut : 64)
parser.add_option("-r", "--learningrate", default=0.005, help="Learning rate", action="store", type="float", dest="learningRate") #Learning Rate (defalut : 0.005)
parser.add_option("-t", "--transform", default='False', help="Transform data size (defalut : False) : if you choose this option, cifar 10 dataset image size will be 227x227", action="store", type="string", dest="transform") #Transform data size (defalut : False)
parser.add_option("-l", "--load", default='False', help="Load model", action="store", type="string", dest="load") #Load Data (defalut : False)
parser.add_option("-o", "--optimizer", default='SGD', help="select optimizer : SGD, Adagrad, Adam ... (Defalut : SGD)", action="store", type="string", dest="optimizer") #Select optimizer (defalut : SGD)
parser.add_option("--patience", default=5, help="Patience for Learning rate scheduler", action="store", type="int", dest="SchedulerPatience") #scheduler patience (default : 5)
parser.add_option("--EarlyStopPatience", default=20, help="patience for Early stop", action="store", type="int", dest="EarlyStopPatience") #early stop patience (default : 20)
parser.add_option("--decay", default=0.005, help="weight decay", action="store", type="float", dest="weightDecay") #weight decay (default : 0.005)

(options, args) = parser.parse_args()


if(options.load == 'True'):
    loadPath = input('\n enter the data path : ')

#Dataset
if options.dataset == 'MNIST':
    dataset = 'MNIST'
    data_dir = './data/MNIST'
else:
    dataset = 'Cifar10'
    data_dir = './data/Cifar10'
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 데이터셋의 평균값
        std=[0.2023, 0.1994, 0.2010]
    )

#Data transfrom
if options.transform == 'True':
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])
else:
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

#data Augmentation
'''
if(options.aug == 'True'):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
'''
#hyper parameters
num_epochs = options.epoch
batch_size = options.batch
learning_rate = options.learningRate
weight_dacay = options.weightDecay
earlystop_patience = options.EarlyStopPatience
scheduler_patience = options.SchedulerPatience

#load data 
train_loader, valid_loader, test_loader, num_classes = get_data(dataset = dataset, data_dir = data_dir ,transform= transform, batch_size = batch_size,random_seed = 1)



#select models
if options.model == 'AlexNet':
    model = AlexNet(num_classes).to(device)
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
elif options.model == 'ResNet18':
    model = ResNet.resnet18(num_classes = num_classes).to(device)
    modelName = 'ResNet18'
elif options.model == 'ResNet34':
    model = ResNet.resnet34(num_classes = num_classes).to(device)
    modelName = 'ResNet34'
elif options.model == 'ResNet50':
    model = ResNet.resnet50(num_classes = num_classes).to(device)
    modelName = 'ResNet50'
elif options.model == 'ResNet101':
    model = ResNet.resnet101(num_classes = num_classes).to(device)
    modelName = 'ResNet101'
elif options.model == 'ResNet152':
    model = ResNet.resnet152(num_classes = num_classes).to(device)
    modelName = 'ResNet152'

#make directory for present data
time = datetime.now()
time = str(time.strftime('%Y-%m-%d %Hh%Mm%Ss'))
folderpath = f'./checkpoint/{modelName}/{time}'
makedirs(folderpath)

#Save terminal log
log_txt = open(folderpath + '/log.txt','w')

train_acc = []
valid_acc = []
'''
#모델 불러오기
PATH = 'c:/Users/hong/workspace/deep_learnig_from_scratch/model_weights.pth'
model = VggNetCifar10(num_classes).to(device)
model.load_state_dict(torch.load(PATH))
'''


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

if options.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay = weight_dacay, momentum = 0.9)
elif options.model == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_dacay, momentum = 0.9)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_dacay, momentum = 0.9)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)  # 학습률 조정 스케줄러
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

printf('Start: ' + time, file= log_txt)
printf('='*50 , file= log_txt)

printf('{}'.format(model), file = log_txt)
printf('='*50 , file= log_txt)
printf('ARG seed : {}'.format(options.seed), file= log_txt)
#print('ARG aug', options.aug, file= log_txt)
printf('ARG dataset: {}'.format(options.dataset), file= log_txt)
printf('ARG model: {}'.format(options.model), file= log_txt)
printf('ARG epoch: {}'.format(options.epoch), file= log_txt)
printf('ARG batch size: {}'.format(options.batch), file= log_txt)
printf('ARG learning rate: {}'.format(options.learningRate), file= log_txt)
printf('ARG transform: {}'.format(options.transform), file= log_txt)
printf('ARG optimizer: {}'.format(options.optimizer), file= log_txt)
printf('ARG patience: {}'.format(options.SchedulerPatience), file= log_txt)
printf('ARG EarlyStopPatience: {}'.format(options.EarlyStopPatience), file= log_txt)
printf('ARG decay: {}'.format(options.weightDecay), file= log_txt)
printf('ARG load: {}'.format(options.load), file= log_txt)

printf('='*50 , file= log_txt)

best_val_acc = 0
# Train the model
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
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

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_acc.append(100 * train_correct / train_total)

    # Validation
    model.eval()
    with torch.no_grad():
        valid_correct = 0
        valid_total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    

            del images, labels, outputs
        valid_acc.append(100 * valid_correct / valid_total)

     # Early Stopping
    if(options.EarlyStop == 'True'):
        if 100 * valid_correct / valid_total > best_val_acc:
            best_val_acc = 100 * valid_correct / valid_total
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= earlystop_patience:
                print('Early stopping')
                break
    
    printf('Model: Epoch [{}/{}] Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()), file= log_txt)
    printf('Accuracy of the Model on the train images: {} %'.format(100 * train_correct / train_total), file= log_txt)
    printf('Accuracy of the Model on the validation images: {} %'.format(100 * valid_correct / valid_total), file= log_txt)
    scheduler.step(valid_correct / valid_total)
    checkpoint_path = folderpath + f'/checkpoint_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    printf('Checkpoint saved at epoch {}\n'.format(epoch+1), file= log_txt)

# Test
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
    
    printf('Accuracy of the Model on the test images: {} %'.format(100 * correct / total), file= log_txt)

time = datetime.now()
time = str(time.strftime('%Y-%m-%d %Hh%Mm%Ss'))
printf('End : ' + time, log_txt)

#Plot
plt.xlabel("epoch")
plt.ylabel("acurracy(%)")
plt.ylim(0, 100)
plt.plot(range(len(train_acc)), train_acc, marker="x", markevery=100, label="train")
plt.plot(range(len(valid_acc)), valid_acc, marker="o",markevery=100, label="valid")
plt.legend()
plt.savefig(folderpath + '/graph.png')
plt.show()
log_txt.close()