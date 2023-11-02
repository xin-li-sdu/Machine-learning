####补全代码，添加注释 2023/10/30 due
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
数据无效项预处理
"""
data = pd.read_csv('./train.csv', encoding = 'utf-8')  #读取训练数据,utf-8，老师提供的big5会报错

data = data.iloc[:, 3:]#去掉前三列无关的文字，只保留数据

data[data == 'NR'] = 0  #将降水有关的NR替换为0，统一数字处理
raw_data = data.to_numpy() #将训练数据转为numpy类型便于处理

month_data = {}  #设置一个空字典来装12个月的数据， 
for month in range(12):     #分为12个月
    sample = np.empty([18, 480]) #存储一个月20天x24 =480个小时的数据，指标18个，所有设置18x480的数组；
    for day in range(20):     #每个月20天
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]#将原始的训练数据按月份和每天进行划分和整理，
        #sample是一个18x480的NumPy数组，用于存储一个月内20天（每天24小时）的数据。其中，18表示该数据集有18个特征（比如温度、湿度、风速等），480表示该数据集总共有480个数据点（每小时一条数据）。
        #具体地，每一列代表一个小时内所有特征的数据，而每一行则代表一个特征在该月内所有小时的数据。
    month_data[month] = sample
    

"""
对于每月，每10h分成一组，由前9h的数据来预测第10h的PM2.5，把前9h的18项指标数据放入x，把第10h的PM2.5数据放入y。
滑动窗口的大小为9，从第1个小时开始向右滑动。故每月都有480-9=471组这样的数据。
""" 
x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) 
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] 


mean_x = np.mean(x, axis = 0) #18 * 9 每一个特征的平均值
std_x = np.std(x, axis = 0) #18 * 9 每一个特征的标准差
for i in range(len(x)): #12 * 471  z-score标准化，通过（x-μ）/σ将两组或多组数据转化为无单位的Z-Score分值，使得数据标准统一化，提高了数据可比性，削弱了数据解释性
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
            
import math
'''
把训练数据分成训练集train_set和验证集validation，其中train_set用于训练，而validation不会参与训练，用于验证。
将数据集按照 8:2 的比例划分为训练集和验证集，
'''
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

'''
在所有的数据的第一列的前面再加一列全为1的属性x_0，系数向量中也增加一个\seta，这样就可以简化y=wx+b为y=x_0*\seta

'''
dim = 18 * 9 + 1  #维度（dim）,18*9个权重(特征数*)+一个常数
w = np.zeros([dim, 1]) 
#x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
x_train_set = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis = 1).astype(float)#train_set用于训练
x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis = 1).astype(float)#看训练效果
iter_time = 4000  #迭代次数
learning_rate = [0.01,0.1,1,10]  #学习率
color_list=['red','green','blue','black','pink','purple']

min_k=0
min_loss=10000   #存储最好的模型（学习率）
eps = 1e-7  #adagrad系数，避免adagrad的分母为0.

for k in range(len(learning_rate)):#Adagrad优化,本部分对于每个学习率进行训练，并用验证集进行验证
    iter=[]
    loss_list=[]#列表用于记录损失函数值随迭代次数变化过程
    w = np.empty([163,1])
    for i in range(len(w)):
        w[i]=[1]
    #loss = np.zeros([iter_time, 1])
    
    adagrad = np.zeros([dim, 1])
   
    for t in range(iter_time):
        #loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/ len(x_train_set))#均方根误差，RMSE
        loss = np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))#MSE
        if t % 10 == 0: #每10次迭代存一次值
            iter.append(t)
            loss_list.append(loss)
        #gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)  #RMSE  
        gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)#MSE
        adagrad += gradient ** 2       #adagrad 是历史梯度平方和的累加  
        w = w - learning_rate[k] * gradient / np.sqrt(adagrad + eps) #w按照学习率和gadient梯度下降
        

    y_pred = np.dot(x_validation, w)
    pred_loss = np.sqrt(np.mean((y_pred - y_validation) ** 2))#rmse
    #pred_loss = np.mean((y_pred - y_validation) ** 2)
    if pred_loss<min_loss:
        min_loss=pred_loss
        min_k=k
    print("Learning Rate = ",learning_rate[k],"\tValidation Loss:", pred_loss)
    plt.plot(iter, loss_list, c=color_list[k],label='learning_rate = '+str(learning_rate[k]))


x = np.concatenate((np.ones([len(x), 1]), x), axis = 1).astype(float)#利用最好的学习率和完整训练集再来一次
loss = np.zeros([iter_time, 1])
best_rate = learning_rate[min_k]
w = np.zeros([dim, 1])
adagrad = np.zeros([dim, 1])
for m in range(iter_time):
    #loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/ len(x))#RMSE
    loss = np.sum(np.power(np.dot(x, w) - y, 2))#MSE
    #gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  #RMSE 
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)#MSE
    adagrad += gradient ** 2         
    w = w - best_rate * gradient / np.sqrt(adagrad + eps)
print("Best Learn Rate =",best_rate,"\tTotal Loss:", loss)

plt.xlabel('iter_times')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()  #绘制图像
np.save('weight.npy', w)

# 导入测试数据test.csv
testdata = pd.read_csv('./test.csv', header = None, encoding = 'utf-8')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans = np.dot(test_x, w)

for i in range(240):
    if(ans[i][0]<0):# 加一个处理,将非0项置零
        ans[i][0]=0
    else:
        ans[i][0]=np.round(ans[i][0])

with open('ans.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans[i][0]]
        csv_writer.writerow(row)
        print(row)
