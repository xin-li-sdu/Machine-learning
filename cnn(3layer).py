# -*- coding: gbk -*-

# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm

#Read image  利用 OpenCV (cv2) 读入照片存放在 numpy array 中
def readfile(path, label):
    # label 是一个 boolean variable
    image_dir = sorted(os.listdir(path))
    # 创建x为输入向量,为输出向量
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)#图片像素大小128*128，3个channel
    y = np.zeros((len(image_dir)), dtype=np.uint8) #每个图片各有1个label
    for i, file in enumerate(image_dir):  #读取图片数据
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))# 改变图像尺寸，统一128*128
        if label:
          y[i] = int(file.split("_")[0])# 判断是否有label，有则存入y
    if label:
      return x, y
    else:
      return x

# 分別把 training set、validation set、testing set 用 readfile 读进来
workspace_dir = './food-11'#定义需要读取的路径
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)#读取训练集数据，参数为True
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)#读取验证集数据，参数为True
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "test"), False)#读取测试集数据，参数为False
print("Size of Testing data = {}".format(len(test_x)))

# Dataset在 PyTorch 中，可以利用 torch.utils.data 的 Dataset 及 DataLoader 来"包裝" data，使后续的 training 及 testing 更方便。

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),#将tensor转换为PIL图像
    transforms.RandomHorizontalFlip(),#随机水平翻转（RandomHorizontalFlip）通过以50%的概率将图像水平翻转来增加图像的多样性，提高模型的泛化能力
    transforms.RandomRotation(15), #随机反转（RandomRotation）通过在一定角度范围内对图像进行随机旋转来增加数据的多样性。转的角度范围是-15到+15度。
    transforms.ToTensor(), #将tensor转换为PIL图像
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),     #将tensor转换为PIL图像                               
    transforms.ToTensor(),#将tensor转换为PIL图像
])
class ImgDataset(Dataset):
#自定义类ImgDataset来继承Dataset，len()返回Dataset大小，getitem()返回某个规定index图片处理后的结果
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label 类型需要是 LongTensor
        self.y = y
        if y is not None:# 将label的类型转为LongTensor
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x) # 返回数据集的大小
    def __getitem__(self, index):
        X = self.x[index]  # 返回数据集的大小
        if self.transform is not None:
            X = self.transform(X)  # 对图片数据进行预处理
        if self.y is not None:
            Y = self.y[index]   # 获取索引为index的标签数据
            return X, Y    # 获取索引为index的标签数据
        else:
            return X  #只返回处理后的图片数据

batch_size =  32   #128  #定义batch大小
#实例化ImgDataset的训练集和验证集
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
#数据加载器是PyTorch中用来封装数据集并使其可迭代的工具。
#将数据集分成小批次，并对数据进行随机化和批处理。在训练过程中，我们需要从数据集中随机抽取一些小批次进行训练，每个小批次包含若干个样本。
#和class ImgDataset结合， getitem()函数获取单个的数据，然后组合成batch，shuffle决定是否打乱数据
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model设置

class Classifier(nn.Module):#深度减半的CNN神经网络
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            #为了达到与模型A相近的参数量，需要适当调整filter的数量
            # ----------------  第一个卷积层+最大池化层  ------------------#
            nn.Conv2d(3, 256, 3, 1, 1), #第一个卷积层，输入通道数为3，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
            nn.BatchNorm2d(256),# 批标准化层，用于对卷积层的输出进行归一化
            nn.ReLU(), # ReLU激活函数
            nn.MaxPool2d(2, 2, 0),# 最大池化层，池化核大小为2x2，步长为2
            # ----------------  第二个卷积层+最大池化层  ------------------#
            nn.Conv2d(256, 512, 3, 1, 1),#第二个卷积层，输入通道数为256，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            nn.BatchNorm2d(512),# 批标准化层，用于对卷积层的输出进行归一化
            nn.ReLU(),# ReLU激活函数
            nn.MaxPool2d(4, 4, 0),# 最大池化层，池化核大小为4x4，步长为4
            # ----------------  第三个卷积层+最大池化层  ------------------#
            nn.Conv2d(512, 512, 3, 1, 1),#第三个卷积层，输入通道数为512，输出通道数为512，卷积核大小为3x3，步长为1，填充为1
            nn.BatchNorm2d(512),#同上，以下不再解释
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
            
        )
        self.fc = nn.Sequential(  #构建全连接神经网络
            # 输入512*4*4，输出1024*1
	        nn.Linear(512*4*4, 1024),#全连接分类器
	        nn.ReLU(),
	        # 输入1024*1，输出512*1
	        nn.Linear(1024, 512),#全连接分类器
	        nn.ReLU(),#激活函数选择ReLU
	        # 输入512*1，输出11*1，即11种食物种类
	        nn.Linear(512, 11)

        )

    def forward(self, x):
        out = self.cnn(x) # 通过卷积层进行前向传播
        out = out.view(out.size()[0], -1) # 将输出展平
        return self.fc(out) # 通过全连接层进行分类

# Training使用 training set 训练，使用 validation set 找好的参数

model = Classifier().cuda()  # 使用CUDA GPU下的pytorch
loss = nn.CrossEntropyLoss() # 因为是 classification task，所以 loss 使用交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 使用 Adam优化器，学习率设置为0.001
num_epoch =  2  #30 # 迭代次数

for epoch in range(num_epoch): 
    epoch_start_time = time.time()# 记录开始时间
    train_acc = 0.0 # 初始化系列准确率
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model 
    for i, data in enumerate(tqdm(train_loader)):#使用tqdm显示进度
        optimizer.zero_grad() # 梯度参数归零，清除之前的梯度信息
        train_pred = model(data[0].cuda()) # 将输入数据传入模型中进行前向计算，得到预测值
        batch_loss = loss(train_pred, data[1].cuda())# 计算这个batch的loss
        batch_loss.backward() # 对损失值进行反向传播，计算每个参数的梯度
        optimizer.step() #根据计算得到的梯度更新模型的参数，以使模型朝着损失函数的最小化方向优化。

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())## 更新准确率
        train_loss += batch_loss.item()# batch loss加入总loss
    
    model.eval()  #将模型设为评估模式，即不进行梯度计算，仅进行前向传播推断
    with torch.no_grad(): # 禁用梯度计算
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())# 将输入数据传递给模型并在CUDA设备上进行计算
            batch_loss = loss(val_pred, data[1].cuda())# 计算损失函数值

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())# 统计正确预测的样本数
            val_loss += batch_loss.item() # 累计批次损失值

        #把結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

#得到好的参数后，使用 training set 和 validation set 共同训练

train_val_x = np.concatenate((train_x, val_x), axis=0)  # 将训练集和验证集的特征数据按行连接在一起
train_val_y = np.concatenate((train_y, val_y), axis=0)  # 将训练集和验证集的标签数据按行连接在一起
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)   # 创建一个包含训练-验证数据集的自定义数据集对象，用于训练和验证
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)  # 使用训练-验证数据集创建一个数据加载器，用于批量加载数据并进行训练

model_best = Classifier().cuda()   # 创建一个基于Classifier类的GPU模型实例
loss = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)   # 创建一个Adam优化器，用于更新模型参数，学习率设置为0.001
#num_epoch = 30  #直接和交叉验证时的次数一样

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(tqdm(train_val_loader)):#使用tqdm显示进度
        optimizer.zero_grad()# 梯度参数归零，清除之前的梯度信息
        train_pred = model_best(data[0].cuda())  # 将输入数据传入模型中进行前向计算，得到预测值
        batch_loss = loss(train_pred, data[1].cuda())  # 计算这个batch的loss
        batch_loss.backward()  # 对损失值进行反向传播，计算每个参数的梯度
        optimizer.step()  #根据计算得到的梯度更新模型的参数，以使模型朝着损失函数的最小化方向优化。

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())## 更新准确率
        train_loss += batch_loss.item()# batch loss加入总loss

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

# Testing利用刚刚 train 好的 model 进行 prediction


test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#把結果写入 csv 
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))