# -*- coding: gbk -*-

# Import��Ҫ���׼�
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

#Read image  ���� OpenCV (cv2) ������Ƭ����� numpy array ��
def readfile(path, label):
    # label ��һ�� boolean variable
    image_dir = sorted(os.listdir(path))
    # ����xΪ��������,Ϊ�������
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)#ͼƬ���ش�С128*128��3��channel
    y = np.zeros((len(image_dir)), dtype=np.uint8) #ÿ��ͼƬ����1��label
    for i, file in enumerate(image_dir):  #��ȡͼƬ����
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))# �ı�ͼ��ߴ磬ͳһ128*128
        if label:
          y[i] = int(file.split("_")[0])# �ж��Ƿ���label���������y
    if label:
      return x, y
    else:
      return x

# �քe�� training set��validation set��testing set �� readfile ������
workspace_dir = './food-11'#������Ҫ��ȡ��·��
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)#��ȡѵ�������ݣ�����ΪTrue
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)#��ȡ��֤�����ݣ�����ΪTrue
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "test"), False)#��ȡ���Լ����ݣ�����ΪFalse
print("Size of Testing data = {}".format(len(test_x)))

# Dataset�� PyTorch �У��������� torch.utils.data �� Dataset �� DataLoader ��"���b" data��ʹ������ training �� testing �����㡣

# training �r�� data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),#��tensorת��ΪPILͼ��
    transforms.RandomHorizontalFlip(),#���ˮƽ��ת��RandomHorizontalFlip��ͨ����50%�ĸ��ʽ�ͼ��ˮƽ��ת������ͼ��Ķ����ԣ����ģ�͵ķ�������
    transforms.RandomRotation(15), #�����ת��RandomRotation��ͨ����һ���Ƕȷ�Χ�ڶ�ͼ����������ת���������ݵĶ����ԡ�ת�ĽǶȷ�Χ��-15��+15�ȡ�
    transforms.ToTensor(), #��tensorת��ΪPILͼ��
])
# testing �r������ data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),     #��tensorת��ΪPILͼ��                               
    transforms.ToTensor(),#��tensorת��ΪPILͼ��
])
class ImgDataset(Dataset):
#�Զ�����ImgDataset���̳�Dataset��len()����Dataset��С��getitem()����ĳ���涨indexͼƬ�����Ľ��
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label ������Ҫ�� LongTensor
        self.y = y
        if y is not None:# ��label������תΪLongTensor
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x) # �������ݼ��Ĵ�С
    def __getitem__(self, index):
        X = self.x[index]  # �������ݼ��Ĵ�С
        if self.transform is not None:
            X = self.transform(X)  # ��ͼƬ���ݽ���Ԥ����
        if self.y is not None:
            Y = self.y[index]   # ��ȡ����Ϊindex�ı�ǩ����
            return X, Y    # ��ȡ����Ϊindex�ı�ǩ����
        else:
            return X  #ֻ���ش�����ͼƬ����

batch_size =  32   #128  #����batch��С
#ʵ����ImgDataset��ѵ��������֤��
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
#���ݼ�������PyTorch��������װ���ݼ���ʹ��ɵ����Ĺ��ߡ�
#�����ݼ��ֳ�С���Σ��������ݽ������������������ѵ�������У�������Ҫ�����ݼ��������ȡһЩС���ν���ѵ����ÿ��С���ΰ������ɸ�������
#��class ImgDataset��ϣ� getitem()������ȡ���������ݣ�Ȼ����ϳ�batch��shuffle�����Ƿ��������
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model����

class Classifier(nn.Module):#��ȼ����CNN������
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input �S�� [3, 128, 128]
        self.cnn = nn.Sequential(
            #Ϊ�˴ﵽ��ģ��A����Ĳ���������Ҫ�ʵ�����filter������
            # ----------------  ��һ�������+���ػ���  ------------------#
            nn.Conv2d(3, 256, 3, 1, 1), #��һ������㣬����ͨ����Ϊ3�����ͨ����Ϊ256������˴�СΪ3x3������Ϊ1�����Ϊ1
            nn.BatchNorm2d(256),# ����׼���㣬���ڶԾ�����������й�һ��
            nn.ReLU(), # ReLU�����
            nn.MaxPool2d(2, 2, 0),# ���ػ��㣬�ػ��˴�СΪ2x2������Ϊ2
            # ----------------  �ڶ��������+���ػ���  ------------------#
            nn.Conv2d(256, 512, 3, 1, 1),#�ڶ�������㣬����ͨ����Ϊ256�����ͨ����Ϊ512������˴�СΪ3x3������Ϊ1�����Ϊ1
            nn.BatchNorm2d(512),# ����׼���㣬���ڶԾ�����������й�һ��
            nn.ReLU(),# ReLU�����
            nn.MaxPool2d(4, 4, 0),# ���ػ��㣬�ػ��˴�СΪ4x4������Ϊ4
            # ----------------  �����������+���ػ���  ------------------#
            nn.Conv2d(512, 512, 3, 1, 1),#����������㣬����ͨ����Ϊ512�����ͨ����Ϊ512������˴�СΪ3x3������Ϊ1�����Ϊ1
            nn.BatchNorm2d(512),#ͬ�ϣ����²��ٽ���
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
            
        )
        self.fc = nn.Sequential(  #����ȫ����������
            # ����512*4*4�����1024*1
	        nn.Linear(512*4*4, 1024),#ȫ���ӷ�����
	        nn.ReLU(),
	        # ����1024*1�����512*1
	        nn.Linear(1024, 512),#ȫ���ӷ�����
	        nn.ReLU(),#�����ѡ��ReLU
	        # ����512*1�����11*1����11��ʳ������
	        nn.Linear(512, 11)

        )

    def forward(self, x):
        out = self.cnn(x) # ͨ����������ǰ�򴫲�
        out = out.view(out.size()[0], -1) # �����չƽ
        return self.fc(out) # ͨ��ȫ���Ӳ���з���

# Trainingʹ�� training set ѵ����ʹ�� validation set �ҺõĲ���

model = Classifier().cuda()  # ʹ��CUDA GPU�µ�pytorch
loss = nn.CrossEntropyLoss() # ��Ϊ�� classification task������ loss ʹ�ý�����
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # ʹ�� Adam�Ż�����ѧϰ������Ϊ0.001
num_epoch =  2  #30 # ��������

for epoch in range(num_epoch): 
    epoch_start_time = time.time()# ��¼��ʼʱ��
    train_acc = 0.0 # ��ʼ��ϵ��׼ȷ��
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # �_�� model ���� train model 
    for i, data in enumerate(tqdm(train_loader)):#ʹ��tqdm��ʾ����
        optimizer.zero_grad() # �ݶȲ������㣬���֮ǰ���ݶ���Ϣ
        train_pred = model(data[0].cuda()) # ���������ݴ���ģ���н���ǰ����㣬�õ�Ԥ��ֵ
        batch_loss = loss(train_pred, data[1].cuda())# �������batch��loss
        batch_loss.backward() # ����ʧֵ���з��򴫲�������ÿ���������ݶ�
        optimizer.step() #���ݼ���õ����ݶȸ���ģ�͵Ĳ�������ʹģ�ͳ�����ʧ��������С�������Ż���

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())## ����׼ȷ��
        train_loss += batch_loss.item()# batch loss������loss
    
    model.eval()  #��ģ����Ϊ����ģʽ�����������ݶȼ��㣬������ǰ�򴫲��ƶ�
    with torch.no_grad(): # �����ݶȼ���
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())# ���������ݴ��ݸ�ģ�Ͳ���CUDA�豸�Ͻ��м���
            batch_loss = loss(val_pred, data[1].cuda())# ������ʧ����ֵ

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())# ͳ����ȷԤ���������
            val_loss += batch_loss.item() # �ۼ�������ʧֵ

        #�ѽY�� print ����
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

#�õ��õĲ�����ʹ�� training set �� validation set ��ͬѵ��

train_val_x = np.concatenate((train_x, val_x), axis=0)  # ��ѵ��������֤�����������ݰ���������һ��
train_val_y = np.concatenate((train_y, val_y), axis=0)  # ��ѵ��������֤���ı�ǩ���ݰ���������һ��
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)   # ����һ������ѵ��-��֤���ݼ����Զ������ݼ���������ѵ������֤
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)  # ʹ��ѵ��-��֤���ݼ�����һ�����ݼ����������������������ݲ�����ѵ��

model_best = Classifier().cuda()   # ����һ������Classifier���GPUģ��ʵ��
loss = nn.CrossEntropyLoss()  # ����һ����������ʧ����
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)   # ����һ��Adam�Ż��������ڸ���ģ�Ͳ�����ѧϰ������Ϊ0.001
#num_epoch = 30  #ֱ�Ӻͽ�����֤ʱ�Ĵ���һ��

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(tqdm(train_val_loader)):#ʹ��tqdm��ʾ����
        optimizer.zero_grad()# �ݶȲ������㣬���֮ǰ���ݶ���Ϣ
        train_pred = model_best(data[0].cuda())  # ���������ݴ���ģ���н���ǰ����㣬�õ�Ԥ��ֵ
        batch_loss = loss(train_pred, data[1].cuda())  # �������batch��loss
        batch_loss.backward()  # ����ʧֵ���з��򴫲�������ÿ���������ݶ�
        optimizer.step()  #���ݼ���õ����ݶȸ���ģ�͵Ĳ�������ʹģ�ͳ�����ʧ��������С�������Ż���

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())## ����׼ȷ��
        train_loss += batch_loss.item()# batch loss������loss

        #���Y�� print ����
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

# Testing���øո� train �õ� model ���� prediction


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

#�ѽY��д�� csv 
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))