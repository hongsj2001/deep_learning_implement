import numpy as np
import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #학습을 gpu가 진행하도록 설정

#Basic Alexnet
'''
1~5번째 레이어는 Convolution layer, 6~8번째 layer는 Fully-connected layer
1,2,5번째 레이어에서 max pooling 수행
모든 레이어에서 Relu 적용
6,7번 레이어 (fully connected의 1,2번째 레이어)에서만  50%의 drop - out 적용
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), #11x11x3 커널을 96개, stride=4로  convolution -> 55x55x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), #LRN 적용
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 27x27x96 feature map 생성
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), #5x5x96 커널을 256개, zero padding = 2로 설정하여 convolution -> 27x27x256 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), #LRN 적용
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 13x13x96 feature map 생성
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 384개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x384 feature map 생성
            nn.ReLU()) #Relu 함수 적용
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 384개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x384 feature map 생성
            nn.ReLU()) #Relu 함수 적용
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 256개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x256 feature map 생성
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 6x6x256 feature map 생성
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(9216, 4096), #input: 9216 -> output : 4096
            nn.ReLU()) #Relu 함수 적용
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(4096, 4096), #input: 9216 -> output : 9216
            nn.ReLU()) #Relu 함수 적용
        self.fc3= nn.Sequential(
            nn.Linear(4096, num_classes) #input: 9216 -> output : 1000
            )
        
    def forward(self, x): #
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, 1) #평탄화 수행
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    


#Alexnet with Cifar-10 (Cifar-10을 데이터셋으로 사용하기 위한 AlexNet 개조 버전)
'''
1~5번째 레이어는 Convolution layer, 6~8번째 layer는 Fully-connected layer
1,2,5번째 레이어에서 max pooling 수행
모든 레이어에서 Relu 적용
6,7번 레이어 (fully connected의 1,2번째 레이어)에서만  50%의 drop-out 적용
'''
class AlexNetCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCifar10, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=0), #11x11x3 커널을 48개, stride=4로  convolution -> 55x55x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), #LRN 적용
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 27x27x96 feature map 생성
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 384, kernel_size=3, stride=1, padding='same'), #5x5x96 커널을 256개, zero padding = 2로 설정하여 convolution -> 27x27x256 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), #LRN 적용
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 13x13x96 feature map 생성
        self.layer3 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=1, padding='same'), #3x3x256 커널을 384개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x384 feature map 생성
            nn.ReLU()) #Relu 함수 적용
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'), #3x3x256 커널을 384개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x384 feature map 생성
            nn.ReLU()) #Relu 함수 적용
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding='same'), #3x3x256 커널을 256개, stride 1, zero padding = 1로 설정하여 convolution -> 13x13x256 feature map 생성
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)) #3x3 커널로 stride=2인  max pooling 수행 -> 6x6x256 feature map 생성
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(512, 256), #input: 9216 -> output : 4096
            nn.ReLU()) #Relu 함수 적용
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(256, 256), #input: 9216 -> output : 9216
            nn.ReLU()) #Relu 함수 적용
        self.fc3= nn.Sequential(
            nn.Linear(256, num_classes) #input: 4096 -> output : 10
            )
        
    def forward(self, x): #
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, 1) #평탄화 수행
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
#Basic VGG-16 
class VggNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(VggNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #3x3x3 커널을 64개, stride=1, zero padding=1로  convolution -> 224x224x64 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 64개, stride=1, zero padding=1로 convolution -> 224x224x64 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2), #2x2 커널로 stride=2인  max pooling 수행 -> 112x112x64 feature map 생성
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 128개, stride=1, zero padding=1로  convolution -> 112x112x128 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 128개, stride=1, zero padding=1로  convolution -> 112x112x128 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2), #2x2 커널로 stride=2인  max pooling 수행 -> 56x56x128 feature map 생성
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #3x3x128 커널을 256개, stride=1, zero padding=1로  convolution -> 56x56x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 256개, stride=1, zero padding=1로  convolution -> 56x56x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 256개, stride=1, zero padding=1로  convolution -> 56x56x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 28x28x256 feature map 생성
            )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 512개, stride=1, zero padding=1로  convolution -> 28x28x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 28x28x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 28x28x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 14x14x512 feature map 생성
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 14x14x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 14x14x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 14x14x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 7x7x512 feature map 생성
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(25088, 4096), #input: 9216 -> output : 4096
            nn.ReLU()) #Relu 함수 적용
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(4096, 4096), #input: 9216 -> output : 9216
            nn.ReLU()) #Relu 함수 적용
        self.fc3= nn.Sequential(
            nn.Linear(4096, num_classes) #input: 9216 -> output : 1000
            )
        
    def forward(self, x): #
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = torch.flatten(out, 1) #평탄화 수행
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    

#VGG-16 with Cifar 10
class VggNetCifar10(nn.Module):
    def __init__(self, num_classes=1000):
        super(VggNetCifar10, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #3x3x3 커널을 64개, stride=1, zero padding=1로  convolution -> 32x32x64 feature map 생성
            nn.ReLU() #Relu 함수 적용
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 64개, stride=1, zero padding=1로 convolution -> 32x32x64 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 128개, stride=1, zero padding=1로  convolution -> 16x16x128 feature map 생성
            nn.ReLU() #Relu 함수 적용
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #3x3x64 커널을 128개, stride=1, zero padding=1로  convolution -> 16x16x128 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 16x16x128 feature map 생성
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #3x3x128 커널을 256개, stride=1, zero padding=1로  convolution -> 8x8x256 feature map 생성
            nn.ReLU() #Relu 함수 적용
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 256개, stride=1, zero padding=1로  convolution -> 8x8x256 feature map 생성
            nn.ReLU() #Relu 함수 적용
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 256개, stride=1, zero padding=1로  convolution -> 8x8x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 8x8x256 feature map 생성
            )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), #3x3x256 커널을 512개, stride=1, zero padding=1로  convolution -> 8x8x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 8x8x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #3x3x512 커널을 512개, stride=1, zero padding=1로 convolution -> 8x8x512 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 2, stride = 2) #2x2 커널로 stride=2인  max pooling 수행 -> 4x4x512 feature map 생성
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(8192, 4096), #input: 9216 -> output : 4096
            nn.ReLU()) #Relu 함수 적용
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), #Drop out (50%) 수행 
            nn.Linear(4096, 4096), #input: 9216 -> output : 9216
            nn.ReLU()) #Relu 함수 적용
        self.fc3= nn.Sequential(
            nn.Linear(4096, num_classes) #input: 9216 -> output : 1000
            )
        
    def forward(self, x): #
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = torch.flatten(out, 1) #평탄화 수행
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out