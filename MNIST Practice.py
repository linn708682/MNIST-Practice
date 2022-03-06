# 1.加載必要的庫
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2.定義超常數

BATCH_SIZE = 16  #每批處理的數據
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #用CPU或GPU來訓練
EPOCHS = 10 #訓練數據集的輪次

# 3.建構pipeline,對圖像做處理
pipeline = transforms.Compose([
    transforms.ToTensor(), #將圖片轉成Tensor
    transforms.Normalize((0.1307,),(0.3081,))]) #正則化:降低模型複雜度 

# 4.下載、加載數據
from torch.utils.data import DataLoader

#下載數據集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline) #下載訓練集

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline) #下載測試集

#加載數據
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #shuffle 把數據集順序打亂

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) #shuffle 把數據集順序打亂

# #顯示MNIST的圖片(補充)
with open("./data/MNIST/raw/train-images-idx3-ubyte","rb") as f:
    file = f.read()

imagel = [int(str(item).encode('ascii'),16)for item in file[16 : 16+784]]
print(imagel)

import cv2
import numpy as np

imagel_np = np.array(imagel,dtype=np.uint8).reshape(28, 28, 1)

print(imagel_np.shape)
cv2.imwrite("digit.jpg", imagel_np)

# 5.建構網路模型

class Digit(nn.Module): #繼承nn.Module類
    def __init__(self): #建構子
        super().__init__()

        #捲積
        self.conv1 = nn.Conv2d(1, 10, 5) #Conv2d(輸入的資料通道數_資料是幾維的, 輸出的資料通道數_經過幾個kernal, kernal的size)
        self.conv2 = nn.Conv2d(10, 20, 3)

        #全連接層(分類器)
        self.fc1 = nn.Linear(20*10*10,500) #Linear(輸入的通道, 輸出的通道)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x): #實現forward方法

        input_size = x.size(0) #batch size

        x = self.conv1(x) #輸入:batch*1*28*28, 輸出:batch*10*24*24  (28-5+1=24)
        x = F.relu(x) #活化層 shape保持不變 輸出:batch*10*24*24
        x = F.max_pool2d(x, 2, 2) #池化層 輸入:batch*10*24*24 輸出:batch*10*12*12 ((24-2)/2+1=12)

        x = self.conv2(x) #輸入:batch*10*12*12 輸出:batch*20*10*10 (12-3+1=10)
        x = F.relu(x)

        x = x.view(input_size, -1) #拉平(為進入全連接層做準備) -1:自動計算維度 (20*10*10=2000)

        x = self.fc1(x) #輸入:batch*2000 輸出:batch*500
        x = F.relu(x)

        x = self.fc2(x) #輸入:batch*500 輸出:batch*10

        output = F.log_softmax(x, dim=1) #計算分類後，每個數字的概率值

        return output

# 6.定義優化器

model = Digit().to(DEVICE) #把模型放到cpu或gpu上

optimizer = optim.Adam(model.parameters()) #optimizer:優化器 更新模型的參數

# 7.定義訓練方法

def train_model(model, device, train_loader, optimizer, epoch):
    #模型訓練
    model.train() #將模型設為訓練模式
    for batch_index, (data, target) in enumerate(train_loader): #train_loader有data, target兩種格式
        #部署到DEVICE上
        data, target = data.to(device), target.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #訓練後的結果
        output = model(data)
        #計算損失
        loss = F.cross_entropy(output, target)
        #反向傳播
        loss.backward()
        #參數優化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

# 8.定義測試方法
def test_model(model, device, test_loader):
    model.eval() #將模型設為測試模式
    #測試損失
    test_loss = 0
    #正確率
    correct = 0
    with torch.no_grad(): #不會計算梯度，也不會進行反向傳播
        for data, target in test_loader:
            #部署到DEVICE上
            data, target = data.to(device), target.to(device)
            #測試數據後的結果
            output = model(data)
            #計算測試損失
            test_loss += F.cross_entropy(output, target) 
            #找到概率值的最大的數字結果
            pred = output.argmax(dim=1, keepdim=True) 
            #累計正確的值
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.item(), correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


# 9.調用方法7/8
for epoch in range(1, EPOCHS +1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
