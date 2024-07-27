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
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2), #3x3 커널로 stride=2인  max pooling 수행 -> 27x27x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)) #LRN 적용
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), #5x5x96 커널을 256개, zero padding = 2로 설정하여 convolution -> 27x27x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2), #3x3 커널로 stride=2인  max pooling 수행 -> 13x13x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)) #LRN 적용
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
    


#Alexnet with Cifar-10
'''
1~5번째 레이어는 Convolution layer, 6~8번째 layer는 Fully-connected layer
1,2,5번째 레이어에서 max pooling 수행
모든 레이어에서 Relu 적용
6,7번 레이어 (fully connected의 1,2번째 레이어)에서만  50%의 drop-out 적용
'''
class AlexNetCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=4, padding=0), #11x11x3 커널을 96개, stride=4로  convolution -> 55x55x96 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding= 1), #3x3 커널로 stride=2인  max pooling 수행 -> 27x27x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)) #LRN 적용
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=2), #5x5x96 커널을 256개, zero padding = 2로 설정하여 convolution -> 27x27x256 feature map 생성
            nn.ReLU(), #Relu 함수 적용
            nn.MaxPool2d(kernel_size = 3, stride = 2), #3x3 커널로 stride=2인  max pooling 수행 -> 13x13x96 feature map 생성
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)) #LRN 적용
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
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, 1) #평탄화 수행 (1x9216)로 변환
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    


class KantapiaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 5)  # 입력 채널 3, 출력 채널 6, 커널 크기 5인 첫 번째 합성곱 층
        self.pool = nn.MaxPool2d(3, 2, 1)  # 커널 크기 2, 스트라이드 2인 최대 풀링 층
        self.conv2 = nn.Conv2d(48, 128, 3, padding='same')  # 입력 채널 6, 출력 채널 16, 커널 크기 5인 두 번째 합성곱 층
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 192, 3, padding='same')
        self.conv4 = nn.Conv2d(192, 192, 3, padding='same')
        self.pool3 = nn.MaxPool2d(3, 2)
        self.conv5 = nn.Conv2d(192, 128, 1, padding='same')
        self.fc1 = nn.Linear(1152, 512)  # 첫 번째 완전 연결 층
        self.fc2 = nn.Linear(512, 256)  # 두 번째 완전 연결 층
        self.fc3 = nn.Linear(256, 10)  # 출력 층 (클래스 개수 10)
        # TODO: padding=same 의 역할 알아오기 : 입/출력의 크기를 동일하도록 패딩 수행
    def forward(self, x):
        x = self.pool(nn.relu(self.conv1(x)))  # 첫 번째 합성곱 후 ReLU 활성화 함수, 풀링 적용
        x = self.pool2(nn.relu(self.conv2(x)))  # 두 번째 합성곱 후 ReLU 활성화 함수, 풀링 적용
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        x = self.pool3(x)
        x = nn.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # 배치 차원 제외 모든 차원을 평탄화
        x = nn.relu(self.fc1(x))  # 첫 번째 완전 연결 층 후 ReLU 활성화 함수 적용
        x = nn.relu(self.fc2(x))  # 두 번째 완전 연결 층 후 ReLU 활성화 함수 적용
        x = self.fc3(x)  # 출력 층
        return x