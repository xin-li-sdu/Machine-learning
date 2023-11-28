# coding=utf-8
import numpy as np
import sys
import csv  
from numpy.linalg import inv
from tqdm import tqdm
from scipy.special import expit

class data_manager():
    def __init__(self):
        self.data = {}  
    
    def read(self,name,path):
        with open(path,newline = '') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:] ,dtype = float) #去掉首行读取
            if name == 'X_train':
                    #求读取的数值均值，转为1行,div=1*106
                    self.mean = np.mean(rows,axis = 0).reshape(1,-1)
                    #求读取的数值求方差，转为1行,div=1*106
                    self.std = np.std(rows,axis = 0).reshape(1,-1)+0.00000000000000001
                    self.theta = np.ones((rows.shape[1] + 1,1),dtype = float)
                    #对每一行数据进行 z-score标准化，（x-μ）/σ
                    for i in range(rows.shape[0]):
                        rows[i,:] = (rows[i,:]-self.mean)/(self.std)
            elif name == 'X_test': 
                    for i in range(rows.shape[0]):
                        rows[i,:] = (rows[i,:]-self.mean)/(self.std)
                #保存数据处理结果到类内字典索引里
            self.data[name] = rows


    def find_theta(self):
        #由于概率生成模型需要计算样本选择的概率，所以需要先对训练数据进行二分类
        class_0_id = []#记录分类的元素索引
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):#通过Y_train分类索引
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)
        #按照索引去X_train找到对应元素
        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id] 
        #分别求出两个分类均值
        mean_0 = np.mean(class_0,axis = 0)
        mean_1 = np.mean(class_1,axis = 0)  

        #不同的类别共享协方差矩阵
        n = class_0.shape[1]
        cov_0 = np.zeros((n,n))
        cov_1 = np.zeros((n,n))

        row_total_num=class_0.shape[0]+class_1.shape[0]
        #for i in range(class_0.shape[0]):#计算类1的协方差
         #   cov_0 += np.dot(np.transpose(class_0[i]-mean_0),class_0[i]-mean_0)
        #for i in range(class_1.shape[0]):#计算类2的协方差
         #   cov_1 += np.dot(np.transpose(class_1[i]-mean_1),class_1[i]-mean_1)
         #cov = (cov_0+cov_1)/row_total_num #两个类共享一个协方差,最终cov就是协方差
        #上面两个循环报错，说协方差矩阵的计算过程中出现了奇异矩阵，改成使用 numpy 提供的函数 np.cov 来计算协方差矩阵

       
        cov_0 = np.cov(class_0, rowvar=False)
        cov_1 = np.cov(class_1, rowvar=False)
        cov = (cov_0 * class_0.shape[0] + cov_1 * class_1.shape[0]) / (row_total_num)
        #两个类共享一个协方差,最终cov就是协方差
        

        #根据公式求w和b
        n_0 = class_0.shape[0]
        n_1 = class_1.shape[0]#类中的行数
        self.w =  np.dot((mean_0-mean_1).transpose(),inv(cov)).transpose()
        self.b =  (-1)*np.dot(np.dot(mean_0.transpose(),inv(cov)),mean_0)/2+np.dot(np.dot(mean_1.transpose(),inv(cov)),mean_1)/2+np.log(float(n_0)/n_1)
        #用求得的w和b计算sigmoid()函数结果
        result = self.func(self.data['X_train'])#经过0-1函数处理得到预测值
        answer = self.predict(result)#计算训练集上的准确率
        accuracy=1-np.mean(np.abs(self.data['Y_train']-answer))
        print("w=",self.w)
        print("b=",self.b)
        print("accuracy = ",accuracy)

    
    #后验概率函数sigmoid
    def func(self,x):
        arr = np.empty([x.shape[0],1],dtype=float)
        ##拆分运算，防止exp溢出，e的幂次方
        for i in range(x.shape[0]):
            z = x[i,:].dot(self.w) + self.b
           # z *= (-1)
            #arr[i][0] = 1 / (1 + np.exp(z))
            arr[i][0] =expit(z)
            #expit()专门针对 logistic sigmoid 函数的计算进行了优化。另外，对于非常大或非常小的输入值，expit(z)在数值上更稳定，避免了溢出或精度丢失的问题。
            #这两个值一个是0.842，一个是0.759，虽然数值不同，但训练出来的模型是一样的，w和b完全一样
        return np.clip(arr, 1e-8, 1-(1e-8))


    def predict(self,x):#函数输出，后验概率值＞0.5则输出1，后验概率值＜0.5则输出0
        ans = np.ones([x.shape[0],1],dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                ans[i] = 0; 
        return ans

    def write_file(self,path):
        result = self.func(self.data['X_test'])
        answer = self.predict(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow(['id','label']) 
            for i in range(answer.shape[0]):
                writer.writerow([i+1,answer[i][0]])
                
                             
dm = data_manager()

dm.read('X_train','data/X_train')#读取文件
dm.read('Y_train','data/Y_train')
dm.read('X_test','data/X_test')
'''
dm.read('X_train','data-test/X_train')
dm.read('Y_train','data-test/Y_train')
dm.read('X_test','data-test/X_test')
'''
dm.find_theta()
dm.write_file('output.csv')
